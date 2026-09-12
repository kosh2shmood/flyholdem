import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .events import Demo, dumps, verify_stream

ROOT = Path(__file__).resolve().parents[3]


def create_app(seed=20260912, replay=None, interval=.9, log_path=None):
    demo = Demo(seed)
    queues = set()
    recorded = [json.loads(line) for line in Path(replay).read_text().splitlines()] if replay else None
    if recorded is not None:
        verify_stream(recorded)
        if not recorded:
            raise ValueError('Replay log is empty')
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
                        if event['kind'] == 'reinforcement':
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

    app = FastAPI(title='FlyHoldem fixture visual prototype', lifespan=lifespan)
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
        return {'ok': bool(task and not task.done()), 'mode': 'fixture', 'replay': bool(recorded)}

    @app.get('/api/graph')
    def graph():
        return demo.brain.graph_view()

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
