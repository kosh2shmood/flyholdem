"""Real Chromium smoke checks, screenshots, and machine-readable results."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8766')
parser.add_argument('--output',default='docs/review')
args=parser.parse_args()
out=Path(args.output)
out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    response=page.goto(args.url)
    assert response.status==200
    page.wait_for_function("window.flyholdem?.latest?.kind === 'decision'", timeout=60000)
    page.get_by_role('button',name='Pause display').click()
    assert 'FIXTURE / VISUAL PROTOTYPE / NOT A FULL-CONNECTOME RESULT' in page.locator('.notice').inner_text()
    assert page.locator('#chosen').inner_text()!='Awaiting spikes'
    assert page.locator('.score').count()==5
    page.wait_for_function("document.querySelectorAll('.comparison-card').length === 3")
    assert page.locator('.comparison-status').all_text_contents()==['Passed','Failed','Passed']
    assert page.locator('.comparison-value').all_text_contents()==['82.3%','47.2%','50.0%','77.8%','54.2%','55.5%','94.5%','55.2%','52.2%']
    assert 'not poker learning results' in page.locator('.control-evidence').inner_text()
    page.locator('.comparison-details summary').nth(2).click()
    assert '128 frozen-model decisions' in page.locator('.comparison-details').nth(2).inner_text()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(out/'fixture-live.png'),full_page=True)
    live=page.evaluate('window.flyholdem.latest')
    page.get_by_role('button',name='Recorded replay',exact=True).click()
    page.wait_for_function("window.flyholdem.mode === 'replay'")
    page.get_by_role('button',name='Resume display').click()
    page.wait_for_function("window.flyholdem?.latest?.kind === 'reinforcement'",timeout=30000)
    page.get_by_role('button',name='Pause display').click()
    assert int(page.locator('#changed').inner_text().replace(',',''))>0
    page.screenshot(path=str(out/'fixture-replay.png'),full_page=True)
    replay=page.evaluate('window.flyholdem.latest')
    page.locator('#input-details summary').first.click()
    assert page.locator('#channels div').count()>0
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(out/'fixture-mobile.png'),full_page=True)
    assert not errors
    assert not page.evaluate('window.flyholdem.errors')
    report={'viewport':[1440,1080],'mobile_viewport':[390,844],
      'live_websocket':True,'replay':True,'visible_weight_changes':True,'exact_input_inspection':True,
      'published_control_comparisons':True,'preserves_failed_transfer':True,'overflow':False,'browser_errors':errors,'live_event_hash':live['hash'],
      'replay_event_hash':replay['hash'],'browser':browser.version,
      'scope':'FIXTURE / VISUAL PROTOTYPE / NOT A FULL-CONNECTOME RESULT'}
    (out/'browser-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    browser.close()
