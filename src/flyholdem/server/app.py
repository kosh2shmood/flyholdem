import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .events import Demo, dumps, verify_stream

ROOT = Path(__file__).resolve().parents[3]


def create_app(seed=20260912, replay=None, interval=.9, log_path=None, mode='fixture', preregistration=None, model=None):
    if mode == 'fixture':
        if model or preregistration:
            raise ValueError('Native model/registration requires circuit or full mode')
        demo = Demo(seed)
    elif mode in ('circuit', 'full'):
        from .native import NativeDemo
        demo = NativeDemo(mode, seed, preregistration, model)
    else:
        raise ValueError('Unknown graph mode')
    fixture_graph = demo.brain.graph_view() if mode == 'fixture' else Demo(seed).brain.graph_view()
    fixture_graph.update(mode='fixture', native_cloud=False, neuron_count=len(fixture_graph['nodes']))
    queues = set()
    recorded = [json.loads(line) for line in Path(replay).read_text().splitlines()] if replay else None
    if recorded is not None:
        verify_stream(recorded)
        if not recorded:
            raise ValueError('Replay log is empty')
        if any(event['mode'] != mode for event in recorded):
            raise ValueError('Recorded graph mode differs from --mode')
        if mode != 'fixture' and any(
                event.get('run_identity', {}).get('graph_sha256') != demo.brain.graph_hash
                or event.get('run_identity', {}).get('registration_sha256') != demo.graph.meta['registration_sha256']
                for event in recorded):
            raise ValueError('Recorded native graph or population registration differs from the viewer')
    latest = None
    log_path = Path(log_path or 'runs/live/events.jsonl')

    @asynccontextmanager
    async def lifespan(app):
        async def produce():
            nonlocal latest
            stream = None
            if recorded is None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open('x', buffering=1)
            try:
                index = 0
                while True:
                    # CPU work off the event loop; exactly one canonical simulation.
                    event = recorded[index % len(recorded)] if recorded else await asyncio.to_thread(demo.next_event)
                    index += 1
                    latest = event
                    if stream:
                        stream.write(dumps(event)+'\n')
                        if event['kind'] in ('reinforcement', 'settlement'):
                            stream.flush()
                            os.fsync(stream.fileno())
                    for queue in tuple(queues):
                        if queue.full():
                            queues.discard(queue)
                            continue
                        queue.put_nowait(event)
                    await asyncio.sleep(interval)
            finally:
                if stream:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
        task = asyncio.create_task(produce())
        app.state.producer = task
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title='FlyHoldem neural poker laboratory', lifespan=lifespan)
    app.mount('/assets', StaticFiles(directory=ROOT/'ui/src'), name='assets')
    vendor = ROOT/'ui/node_modules/three'
    if vendor.is_dir():
        app.mount('/vendor/three', StaticFiles(directory=vendor), name='three')

    @app.get('/')
    def index():
        return FileResponse(ROOT/'ui/index.html')

    @app.get('/api/health')
    def health():
        task = getattr(app.state, 'producer', None)
        return {'ok': bool(task and not task.done()), 'mode': mode, 'replay': bool(recorded), 'weights_frozen': mode != 'fixture'}

    @app.get('/api/graph')
    def graph(mode: str | None = None):
        if mode == 'fixture':
            return fixture_graph
        current = fixture_graph if not hasattr(demo, 'graph') else demo.graph.meta
        if mode is not None and mode != current['mode']:
            raise HTTPException(404, 'Graph is not loaded in this server')
        return current

    @app.get('/api/graph/{buffer_name}')
    def graph_buffer(buffer_name: str):
        if buffer_name not in ('positions', 'roles') or not hasattr(demo, 'graph'):
            raise HTTPException(404, 'Native graph buffer unavailable')
        return Response(getattr(demo.graph, buffer_name).tobytes(), media_type='application/octet-stream')

    @app.get('/api/graph/neuron/{index}')
    def neuron(index: int):
        if not hasattr(demo, 'graph'):
            raise HTTPException(404, 'Native annotation unavailable')
        try:
            return demo.graph.node(index)
        except IndexError:
            raise HTTPException(404, 'Neuron index out of range')

    @app.get('/api/evidence')
    def evidence():
        import yaml
        return yaml.safe_load((ROOT/'configs/evidence.yaml').read_text())

    @app.get('/api/example')
    def example():
        return FileResponse(ROOT/'examples/fixture-demo.jsonl', media_type='application/x-ndjson')

    @app.websocket('/ws')
    async def websocket(ws: WebSocket):
        await ws.accept()
        queue = asyncio.Queue(maxsize=64)
        queues.add(queue)
        if latest:
            queue.put_nowait(latest)
        try:
            while True:
                event = await queue.get()
                await ws.send_text(dumps(event))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            queues.discard(queue)
    return app
