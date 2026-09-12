"""Verify native full/circuit rendering, real neural events and replay switching."""
import argparse
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:8767');p.add_argument('--output',default='runs/native-browser-check')
args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1200})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda e:errors.append(e.text) if e.type=='error' else None)
    page.goto(args.url)
    page.wait_for_function('window.flyholdem?.brainCloud?.snapshot().drawn_points>0',timeout=60000)
    page.wait_for_function('''() => {if(window.flyholdem?.latest?.kind!=="decision")return false;document.querySelector("#pause").click();return true}''',timeout=60000)
    state=page.evaluate('({event:window.flyholdem.latest,graph:window.flyholdem.graph,cloud:window.flyholdem.brainCloud.snapshot()})')
    native_mode=state['graph']['mode'];n=state['graph']['neuron_count']
    assert state['event'].get('decision') and state['cloud']['activity_total']==state['event']['decision']['activity_total']
    assert state['event']['mode']==native_mode and native_mode in ('circuit','full')
    assert state['cloud']['neurons']==state['cloud']['drawn_points']==n
    assert state['cloud']['paused'] is True
    assert state['event']['teacher']=='disconnected' and state['event']['plasticity_enabled'] is False
    for name in ('positions','roles'):
        body=page.request.get(args.url+'/api/graph/'+name).body()
        assert hashlib.sha256(body).hexdigest()==state['graph'][name+'_sha256']
    assert page.request.get(args.url+'/api/graph/neuron/0').json()['body_id']
    assert page.request.get(args.url+'/api/graph/neuron/-1').status==404
    assert 'POKER POLICY UNVALIDATED' in page.locator('#run-label').inner_text()
    assert page.locator('#plasticity-mode').inner_text()=='FROZEN WEIGHTS'
    assert page.locator('#dopamine').inner_text()=='Not delivered'
    page.locator('#brain-filter').select_option('2')
    assert page.evaluate('window.flyholdem.brainCloud.snapshot().filter')==2
    page.locator('#brain-filter').select_option('0')
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(out/'native-full-dashboard.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(out/'native-mobile.png'),full_page=True)
    page.set_viewport_size({'width':1440,'height':1200})
    page.get_by_role('button',name='Recorded replay',exact=True).click()
    page.get_by_role('button',name='Resume display').click()
    page.wait_for_function('window.flyholdem?.latest?.mode==="fixture" && window.flyholdem?.graph?.mode==="fixture" && !window.flyholdem.brainCloud',timeout=60000)
    assert 'FIXTURE / VISUAL PROTOTYPE' in page.locator('#run-label').inner_text()
    assert page.locator('#brain-native').is_hidden()
    page.get_by_role('button',name='Live',exact=True).click()
    page.wait_for_function('window.flyholdem?.latest?.mode!=="fixture" && window.flyholdem?.brainCloud?.snapshot().drawn_points>0',timeout=60000)
    page.wait_for_function('''() => {if(window.flyholdem?.latest?.kind!=="decision")return false;document.querySelector("#pause").click();return true}''',timeout=60000)
    final=page.evaluate('({event:window.flyholdem.latest,cloud:window.flyholdem.brainCloud.snapshot(),errors:window.flyholdem.errors})')
    if final['event'].get('decision'):
        assert final['cloud']['activity_total']==final['event']['decision']['activity_total']
    assert not errors and not final['errors']
    report={'status':'pass','mode':native_mode,'retained_neurons_rendered':n,
        'native_source':'MaleCNS-native-LIF-spikes','frozen_evaluation':True,'teacher_disconnected':True,
        'exact_geometry_checksums':True,'annotation_endpoint':True,'filters':True,'pause':True,
        'fixture_replay_then_native_live':True,'mobile_no_overflow':True,'console_and_page_errors':errors,
        'located_neurons':state['graph']['located_neurons'],'unlocated_grid_neurons':state['graph']['unlocated_neurons'],
        'browser':browser.version,'event_hash':state['event']['hash']}
    (out/'native-browser-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));browser.close()
