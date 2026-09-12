"""Exercise the real heads-up UI: private cards, legality, reload and gestures."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:8766')
parser.add_argument('--output',default='runs/human-browser-check')
args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1120})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda e:errors.append(e.text) if e.type=='error' else None)
    page.goto(args.url);page.wait_for_function('window.flyholdem?.avatar && window.flyholdem?.latest',timeout=60000)
    page.locator('#play').click()
    def idle():page.wait_for_function('window.flyholdem.mode==="play" && !document.querySelector("#play").disabled && window.flyholdem.latest.viewer==="human"',timeout=60000)
    idle()
    checked=0;actions=0;seen=set();hands=set();moved=set()
    for step in range(35):
        state=page.evaluate('({event:window.flyholdem.latest,avatar:window.flyholdem.avatar.snapshot(),cards:[...document.querySelectorAll("#avatar-cards .card")].map(x=>x.dataset.card),board:[...document.querySelectorAll("#avatar-community-cards .card")].map(x=>x.dataset.card),token:sessionStorage.getItem("flyholdem-play-session")})')
        e=state['event'];t=e['table'];hands.add(e['hand']);seen.add(len(t['board']))
        assert e.get('decision') is None and e['private_diagnostics_withheld']
        assert state['cards']==t['opponent_hole'] and all(v!='??' for v in state['cards'])
        assert state['board']==t['board']+['']*(5-len(t['board']))
        assert state['avatar']['hole']==t['hole'] and state['avatar']['opponent']['hole']==t['opponent_hole']
        if not t['done']:assert t['hole']==['??','??']
        for key in ('scores','encoded_hash','private_hand','initial_deck','activity_counts'):
            assert key not in json.dumps(e)
        for selector in ('.decision-panel','.brain-panel','.plasticity-panel'):
            assert page.locator(selector).is_hidden()
        assert page.locator('#pause').is_disabled()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if state['avatar']['action_frames']>0:moved.add('fly')
        if state['avatar']['opponent']['action_frames']>0:moved.add('human')
        checked+=1
        if checked==1:
            token=state['token'];seq=e['sequence'];page.reload();idle()
            assert page.evaluate('sessionStorage.getItem("flyholdem-play-session")')==token
            assert page.evaluate('window.flyholdem.latest.sequence')==seq
            page.wait_for_function('window.flyholdem.avatar')
            restored=page.evaluate('window.flyholdem.avatar.snapshot()')
            for seat,key in ((0,None),(1,'opponent')):
                history=[a for a in t['history'] if a['actor']==seat]
                if history:assert (restored if key is None else restored[key])['action']==history[-1]['action']
            assert page.request.get(args.url+'/api/health').json()['spectator_paused_for_play']
            page.screenshot(path=str(out/'human-desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            page.screenshot(path=str(out/'human-mobile.png'),full_page=True)
            page.set_viewport_size({'width':1440,'height':1120})
        if t['done']:
            if len(hands)>=6 and moved=={'fly','human'}:break
            page.locator('#play-next').click();idle();continue
        choices=page.locator('#play-actions button').evaluate_all('(xs)=>xs.map(x=>({id:+x.dataset.action,disabled:x.disabled}))')
        legal=[v['id'] for v in choices if not v['disabled']]
        assert legal
        # Fixed nonstrategic browser-check policy, never a fly fallback.
        action=1 if 1 in legal else legal[0]
        page.locator(f'#play-actions button[data-action="{action}"]').click();idle();actions+=1
    assert actions>0 and moved=={'fly','human'}
    token=page.evaluate('sessionStorage.getItem("flyholdem-play-session")')
    page.locator('#live').click()
    page.wait_for_function('window.flyholdem.mode==="live" && window.flyholdem.latest.viewer!=="human"',timeout=60000)
    assert page.request.get(args.url+'/api/play/'+token).status==410
    assert not page.request.get(args.url+'/api/health').json()['spectator_paused_for_play']
    assert page.locator('.decision-panel').is_visible()
    assert not errors and not page.evaluate('window.flyholdem.errors')
    report={'status':'pass','graph':e['mode'],'checked_states':checked,'hands':len(hands),'human_actions':actions,
        'board_counts_observed':sorted(seen),'motion_players':sorted(moved),'private_cards_and_diagnostics':True,
        'reload_retains_hand':True,'server_legal_controls':True,'leave_resumes_spectator':True,
        'desktop_and_mobile_no_overflow':True,'browser':browser.version,'errors':errors}
    (out/'human-browser-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));browser.close()
