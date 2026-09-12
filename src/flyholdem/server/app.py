import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
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
    play_session = None
    play_opening = False
    play_lock = asyncio.Lock()
    spectator_idle = asyncio.Event(); spectator_idle.set()
    log_path = Path(log_path or 'runs/live/events.jsonl')

    @asynccontextmanager
    async def lifespan(app):
        async def produce():
            nonlocal latest, play_session
            stream = None
            if recorded is None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open('x', buffering=1)
            try:
                index = 0
                while True:
                    if play_opening or play_session is not None:
                        if play_session is not None and time.monotonic()-play_session.last_access>1800 and not play_lock.locked():
                            async with play_lock:
                                await asyncio.to_thread(play_session.close);play_session=None
                        await asyncio.sleep(.1)
                        continue
                    # CPU work off the event loop; exactly one canonical simulation.
                    spectator_idle.clear()
                    try:
                        event = recorded[index % len(recorded)] if recorded else await asyncio.to_thread(demo.next_event)
                    finally:
                        spectator_idle.set()
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
        if play_session is not None:
            play_session.close()

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
        return {'ok': bool(task and not task.done()), 'mode': mode, 'replay': bool(recorded), 'weights_frozen': mode != 'fixture', 'spectator_paused_for_play': play_opening or play_session is not None}

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

    def check_origin(request):
        origin=request.headers.get('origin')
        if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
            raise HTTPException(403,'Play requests must come from this local dashboard')

    async def play_payload(request,fields):
        check_origin(request)
        try: value=await request.json()
        except Exception: raise HTTPException(422,'A JSON action request is required')
        if not isinstance(value,dict) or set(value)!=set(fields):
            raise HTTPException(422,'Unexpected play request fields')
        return value

    def active_session(token):
        if play_session is None or token!=play_session.token:
            raise HTTPException(410,'This match has expired. Start a new match.')
        return play_session

    @app.post('/api/play/start')
    async def start_play(request: Request):
        nonlocal play_session,play_opening
        check_origin(request)
        from .play import FrozenPlayer,HumanSession,PlayError
        async with play_lock:
            play_opening=True
            try:
                # Finish the spectator's current operation before allocating a
                # private neural player. Never advance two full workers together.
                await spectator_idle.wait()
                if play_session is not None:
                    await asyncio.to_thread(play_session.close);play_session=None
                player=await asyncio.to_thread(FrozenPlayer,demo)
                play_session=HumanSession(player,log_path.parent/'play-private')
                return await asyncio.to_thread(play_session.new_hand)
            except PlayError as error:
                raise HTTPException(error.status,str(error))
            finally:
                play_opening=False

    @app.get('/api/play/{token}')
    async def play_state(token: str):
        async with play_lock:
            return active_session(token).response()

    @app.post('/api/play/{token}/action')
    async def play_action(token: str,request: Request):
        from .play import PlayError
        value=await play_payload(request,('revision','action'))
        async with play_lock:
            try:
                return await asyncio.to_thread(active_session(token).act,value['revision'],value['action'])
            except PlayError as error:
                raise HTTPException(error.status,str(error))

    @app.post('/api/play/{token}/hand')
    async def play_hand(token: str,request: Request):
        from .play import PlayError
        value=await play_payload(request,('revision',))
        async with play_lock:
            try:
                return await asyncio.to_thread(active_session(token).new_hand,value['revision'])
            except PlayError as error:
                raise HTTPException(error.status,str(error))

    @app.post('/api/play/{token}/end')
    async def end_play(token: str,request: Request):
        nonlocal play_session
        check_origin(request)
        async with play_lock:
            session=active_session(token)
            await asyncio.to_thread(session.close);play_session=None
            return {'closed':True}

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
