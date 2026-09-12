import argparse
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='FlyHoldem play-chip research tools')
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve', help='Run the labeled live or recorded fixture dashboard')
    serve.add_argument('--port', type=int, default=8766)
    serve.add_argument('--seed', type=int, default=20260912)
    serve.add_argument('--replay')
    serve.add_argument('--interval', type=float, default=.9)
    record = sub.add_parser('record', help='Generate a deterministic compact fixture demonstration')
    record.add_argument('--hands', type=int, default=6)
    record.add_argument('--seed', type=int, default=20260912)
    record.add_argument('--output', default='examples/fixture-demo.jsonl')
    args = parser.parse_args()
    if args.command == 'serve':
        import uvicorn
        from flyholdem.server.app import create_app
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        uvicorn.run(create_app(args.seed, args.replay, args.interval,
            f'runs/live-{stamp}/events.jsonl'), host='127.0.0.1', port=args.port)
    elif args.command == 'record':
        from flyholdem.server.events import Demo, dumps
        demo = Demo(args.seed)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as stream:
            completed = 0
            while completed < args.hands:
                event = demo.next_event()
                stream.write(dumps(event)+'\n')
                completed += event['kind'] == 'reinforcement'
        print(f'{path}: {args.hands} fixture hands, {demo.sequence} hash-chained events')


if __name__ == '__main__':
    main()
