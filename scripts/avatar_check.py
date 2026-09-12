"""Verify every gesture against actual checked-in poker events in Chromium."""
import json
import math
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

out=Path('docs/review/avatar');out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch()
    context=browser.new_context(viewport={'width':1440,'height':1080},device_scale_factor=1,
        record_video_dir='work/avatar-video',record_video_size={'width':1280,'height':960})
    page=context.new_page()
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8766')
    page.wait_for_function('window.flyholdem?.avatar?.snapshot().triangles > 0',timeout=30000)
    page.wait_for_function('window.flyholdem?.latest?.kind === "decision"',timeout=30000)
    live=page.evaluate('({event:window.flyholdem.latest,avatar:window.flyholdem.avatar.snapshot()})')
    assert live['event']['hash']==live['avatar']['event_hash']
    assert live['event']['table']['hole']==live['avatar']['hole']
    page.get_by_role('button',name='Recorded replay',exact=True).click()
    page.wait_for_function('window.flyholdem.mode === "replay" && window.flyholdem.latest.sequence === 0')
    assert page.locator('#changed').inner_text()=='—', 'Replay must clear previous live plasticity'
    initial=page.evaluate('window.flyholdem.avatar.snapshot()')
    assert initial['position'][0]<0<initial['opponent']['position'][0]
    for card in initial['cards']:
        assert card['bend']>0 and card['holder_facing']>0 and card['viewer_facing']>0
        assert card['printed_side']=='front-only' and card['back']=='pattern'
    assert initial['opponent']['hole']==['??','??']
    natural_cards=initial['cards']
    found={};opponent_found={};board_lengths=set();deadline=time.monotonic()+65
    while time.monotonic()<deadline and (len(found)<6 or len(opponent_found)<2):
        state=page.evaluate('({event:window.flyholdem.latest,avatar:window.flyholdem.avatar.snapshot()})')
        a,e=state['avatar'],state['event']
        opponent=a['opponent']
        board_lengths.add(len(e['table']['board']))
        # The browser may advance between tool reads; compare within one JS task.
        inset=page.evaluate("({board:window.flyholdem.latest.table.board,shown:[...document.querySelectorAll('#avatar-community-cards .card')].map(n=>n.dataset.card).filter(Boolean)})")
        assert inset['board']==inset['shown']
        assert (a['visible_chip_count']>0)==(e['table']['stacks'][0]>0)
        assert (opponent['visible_chip_count']>0)==(e['table']['stacks'][1]>0)
        assert opponent['hole']==e['table']['opponent_hole'], 'Never reveal private opponent cards'
        if e['kind']=='opponent_action' and .15<opponent['elapsed']<.72:
            label='check' if opponent['check'] else 'call'
            if label not in opponent_found:
                assert opponent['action_hash']==e['hash'] and opponent['action']==e['action']['action']
                assert opponent['paid']==e['action']['paid'] and opponent['action_frames']>0
                displacement=math.dist(opponent['right'],[.43,1.13,-.27])
                assert displacement>.025
                opponent_found[label]={'event_hash':e['hash'],'sequence':e['sequence'],
                    'right_displacement':displacement,'action_frames':opponent['action_frames']}
                page.get_by_role('button',name='Pause display').click()
                page.locator('#avatar-stage').screenshot(path=str(out/f'opponent-{label}.png'))
                before=page.evaluate('window.flyholdem.avatar.snapshot().opponent')
                page.wait_for_timeout(150)
                after=page.evaluate('window.flyholdem.avatar.snapshot().opponent')
                assert before['left']==after['left'] and before['right']==after['right'] and before['elapsed']==after['elapsed']
                page.get_by_role('button',name='Resume display').click()
        label='check' if a['check'] else {0:'fold',1:'call',2:'half-pot',3:'pot',4:'all-in'}.get(a['action'])
        if label and label not in found and e['kind']=='decision' and .15<a['elapsed']<.72:
            assert a['action_hash']==e['hash']
            assert a['action']==e['decision']['selected']
            assert a['hole']==e['table']['hole']
            assert a['board']==e['table']['board']
            assert a['action_frames']>0
            left_delta=math.dist(a['left'],[-.43,1.13,-.27])
            right_delta=math.dist(a['right'],[.43,1.13,-.27])
            assert max(left_delta,right_delta)>.025
            if label=='all-in':
                assert left_delta>.1 and right_delta>.1
            found[label]={'event_hash':e['hash'],'sequence':e['sequence'],'hand':e['hand'],
                'left_displacement':left_delta,'right_displacement':right_delta,'cards':a['hole'],
                'action_frames':a['action_frames'],'elapsed':a['elapsed']}
            page.get_by_role('button',name='Pause display').click()
            page.screenshot(path=str(out/f'fly-{label}.png'),full_page=True)
            if label=='all-in':
                page.locator('#avatar-stage').screenshot(path=str(out/'fly-closeup.png'))
                before=page.evaluate('window.flyholdem.avatar.snapshot()')
                page.wait_for_timeout(200)
                after=page.evaluate('window.flyholdem.avatar.snapshot()')
                assert before['left']==after['left'] and before['elapsed']==after['elapsed']
            page.get_by_role('button',name='Resume display').click()
        page.wait_for_timeout(40)
    assert len(found)==6, f'Missing real gesture coverage: {found.keys()}'
    assert {0,3,5}.issubset(board_lengths), board_lengths
    assert len(opponent_found)==2, f'Missing opponent gestures: {opponent_found.keys()}'
    page.get_by_role('button',name='Pause display').click()
    page.get_by_role('button',name='Table view',exact=True).click()
    assert page.locator('#table-map').is_visible() and not page.locator('#avatar-stage').is_visible()
    page.get_by_role('button',name='3D fly',exact=True).click()
    assert page.locator('#avatar-stage').is_visible()
    page.get_by_role('button',name='Reset view').click()
    def visible_models():
        state=page.evaluate('window.flyholdem.avatar.snapshot()')
        for player in [state,state['opponent']]:
            bounds=player['screen_bounds']
            assert -1<bounds['left']<bounds['right']<1 and -1<bounds['bottom']<bounds['top']<1, bounds
        return {key:player['screen_bounds'] for key,player in [('fly',state),('opponent',state['opponent'])]}
    page.wait_for_timeout(100)
    desktop_bounds=visible_models()
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_timeout(300)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    mobile_bounds=visible_models()
    page.screenshot(path=str(out/'fly-mobile.png'),full_page=True)
    assert not errors and not page.evaluate('window.flyholdem.errors')
    snapshot=page.evaluate('window.flyholdem.avatar.snapshot()')
    report={'status':'pass','scope':'illustrative action-driven avatar, not a body simulation',
        'live_event_card_match':True,'replay_event_card_match':True,'six_gestures':found,'opponent_gestures':opponent_found,
        'curved_holder_and_viewer_facing_cards':natural_cards,'opponent_information_boundary':True,'community_inset_matches_events':True,
        'community_card_counts_seen':sorted(board_lengths),'both_chip_stacks_match_zero_or_positive_balance':True,
        'full_silhouette_bounds':{'desktop':desktop_bounds,'mobile':mobile_bounds},
        'pause_freezes_both_players':True,'camera_reset':True,'table_toggle':True,
        'desktop':[1440,1080],'mobile':[390,844],'browser':browser.version,
        'webgl':snapshot['renderer'],'three_revision':snapshot['revision'],
        'triangles':snapshot['triangles'],'draw_calls':snapshot['draw_calls'],'errors':errors}
    video=page.video
    context.close()
    video.save_as(str(out/'fly-action-demo.webm'))
    browser.close()
    (out/'avatar-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
