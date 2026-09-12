import {renderEvidenceComparisons} from '/assets/evidence-comparisons.js';
const $ = id => document.getElementById(id);
const names = ['Fold', 'Check / call', 'Raise ½ pot', 'Raise pot', 'All-in'];
const suits = {c:'♣', d:'♦', h:'♥', s:'♠'};
let socket, mode='live', paused=false, latest=null, timer, replayEvents=[], replayIndex=0, graph=null;
let activity=Array(126).fill(0), shown=Array(126).fill(0);
let trainingServer=false;
let playSession=null,playRevision=-1,playEpoch=0,playBusy=false,playHeartbeat=null;
let decisionHand=null, avatar=null, brainCloud=null, graphLoading=null, spikeTotal=0;
const errors=[];
window.addEventListener('error', e => errors.push(e.message));
window.flyholdem = {get latest(){return latest}, get mode(){return mode}, get avatar(){return avatar}, get graph(){return graph}, get brainCloud(){return brainCloud}, errors};
function card(value){
 const el=document.createElement('span'); el.className='card';el.dataset.card=value||'';
 if(value==='??'){el.classList.add('back');el.textContent='◇';return el}
 if(!value){el.classList.add('empty');return el}
 if(!/^[2-9TJQKA][cdhs]$/.test(value))throw Error('Invalid event card');
 if('dh'.includes(value[1]))el.classList.add('red');
 el.append(document.createTextNode(value[0]==='T'?'10':value[0]));
 const suit=document.createElement('span');suit.className='suit';suit.textContent=suits[value[1]];el.append(suit);return el;
}
function cards(id, values, count){const target=$(id);target.replaceChildren();for(let i=0;i<count;i++)target.append(card(values[i]||''))}
for(let i=0;i<5;i++){
 const row=document.createElement('div');row.className='score';row.id=`score-${i}`;
 row.innerHTML=`<div class="score-label"><span class="mask">—</span><span>${names[i]}</span><span class="score-value">—</span></div><div class="bar"><i></i></div>`;
 $('scores').append(row);
}
function resetDecision(){
 decisionHand=null;$('chosen').textContent='Awaiting spikes';$('decision-context').textContent='A fresh hand. Waiting for the fly.';
 $('channels').replaceChildren();$('observation').textContent='';$('input-hash').textContent='—';$('silent').textContent='—';$('temperature').textContent='—';
 for(let i=0;i<5;i++){const row=$(`score-${i}`);row.className='score';row.querySelector('.mask').textContent='—';row.querySelector('.score-value').textContent='—';row.querySelector('i').style.width='0'}
}
function resetPlasticity(){
 $('recorded-pulse-note').hidden=true;$('recorded-pulse-note').textContent='';
 $('dopamine').textContent='—';$('changed').textContent='—';$('weight-delta').textContent='—';$('eligibility').textContent='—';$('ratios').textContent='—';$('pulse').style.width='0';activity=Array(126).fill(0);shown=Array(126).fill(0);
}
function render(event){
 latest=event;
 const human=event.viewer==='human';document.body.dataset.viewer=human?'human':'spectator';
 const opponentNames={'random':'Random','calling-station':'Calling station','tight-aggressive':'Tight-aggressive','equity-bucket':'Equity-bucket'};
 const otherName=human?'You':opponentNames[event.opponent]||(event.opponent?.startsWith('frozen-native:')?'Frozen native opponent':'Station');
 const fixtureMode=event.mode.startsWith('fixture');
 const playedHistory=event.table.history.slice(event.setup_action_count||0);
 $('intro-copy').textContent=event.recorded_training?'An audited training replay. Cards, neural scores and recorded activity follow the saved hands.':'A live, inspectable loop connecting poker information, neural activity and legal actions.';
 $('table-stakes').textContent=`TABLESIDE CAM · ${event.stack_bb||20} BB / PLAY CHIPS`;
 $('source').textContent=event.recorded_training?'TRAINING RECORD':human?'YOUR PRIVATE MATCH':mode==='replay'?'CHECKED-IN REPLAY':'LIVE STREAM';
 $('return-label').textContent=human?'FLY’S MATCH RESULT':event.recorded_training?'FLY’S RECORDED HANDS':'FLY’S DEMO RESULT';
 $('fly-seat-name').textContent=fixtureMode?'FIXTURE FLY':'NATIVE FLY';
 $('table-caption').textContent=human?'Your cards are visible. Fly cards appear only if shown at showdown.':'Opponent cards hidden until revealed at showdown';
 $('hand-inset-caption').textContent=human?'YOUR HAND':'FLY’S HAND · FACE UP FOR YOU';
 $('avatar-opponent-name').textContent=human?'YOU':(event.opponent?otherName.toUpperCase():'CALLING STATION');$('opponent-seat-name').textContent=human?'YOU':(event.opponent?otherName.toUpperCase():'CALLING STATION');$('other-stack-label').textContent=human?'YOU':'OPPONENT';
 if(graph?.mode!==event.mode)ensureGraph(event.mode).catch(e=>errors.push(e.message));
 $('run-label').textContent=event.label;
 $('run-scope').textContent=fixtureMode?'Play chips only · No demonstrated learning':'Play chips only · No validated poker learning';
 $('mode-graph').textContent=event.mode.toUpperCase()+' GRAPH';$('mode-learning').textContent=event.learning_mode;
 $('mode-teacher').textContent=event.teacher==='connected'?'Teacher connected · recorded training':'Teacher disconnected';
 $('mode-status').textContent=event.recorded_training?`${event.status} · Recorded training · ${event.arm}`:event.evaluation?'Frozen evaluation · Development':event.status;
 $('plasticity-mode').textContent=event.plasticity_enabled===false?'FROZEN WEIGHTS':event.optimization==='direct-readout-rate-surrogate-v1'?'NONBIOLOGICAL SURROGATE':'LOCAL RULE';
 $('plasticity-note').textContent=event.plasticity_enabled===false?'Learning is disabled. Chip outcomes are recorded; no dopamine pulse or synaptic update is delivered.':event.optimization==='direct-readout-rate-surrogate-v1'?'Recorded bounded updates on existing edges use a nonbiological gradient approximation. No dopamine teaching pulse is delivered.':'Dopamine × eligibility × learning rate. Weight movement is a mechanism check, not evidence of learning.';
 if(event.plasticity_enabled===false){$('dopamine').textContent='Not delivered';$('changed').textContent='0';$('weight-delta').textContent='0';$('eligibility').textContent='Disabled';$('ratios').textContent='Frozen model';$('pulse').style.width='0'}
 if(event.sequence===0){resetPlasticity();spikeTotal=0;brainCloud?.setActivity([],[])}
 if(human)$('event-counter').textContent=`EVENT ${event.sequence} · PRIVATE MATCH`;
 avatar?.update({...event,table:{...event.table,history:playedHistory}});
 cards("avatar-cards",human?event.table.opponent_hole:event.table.hole,2);cards("avatar-community-cards",event.table.board,5);$("avatar-board-stage").textContent=event.table.street;
 $("avatar-stack-fly").textContent=event.table.stacks[0];$("avatar-stack-other").textContent=event.table.stacks[1];$("avatar-pot").textContent=event.table.pot;
 if(event.kind==="hand_start"){$("avatar-action").textContent="Looking at its cards";$("avatar-motion").textContent="A new hand. Waiting for neural output."}
 const flyAction=[...playedHistory].reverse().find(a=>a.actor===0);
 if(flyAction){
  const d={selected:flyAction.action,observation:{to_call:flyAction.paid}};$("avatar-action").textContent=names[d.selected];
  $("avatar-motion").textContent=d.selected===0?"Slides both cards into the muck":d.selected===1?(d.observation.to_call===0?"Taps the felt to check":"Pushes chips forward to call"):d.selected===4?"Both hands push the stack in":"Reaches forward with a raise";
 }
 if(event.kind==='hand_start'){$('avatar-opponent-action').textContent='Looking at its cards';$('avatar-opponent-motion').textContent='Waiting for its turn';resetDecision()}
 const otherAction=[...playedHistory].reverse().find(a=>a.actor===1);
 if(otherAction){
  const a=otherAction;$('avatar-opponent-action').textContent=a.action===1?(a.paid?'Call':'Check'):names[a.action];
  $('avatar-opponent-motion').textContent=a.action===0?'Slides its cards away':a.action===1?(a.paid?'Pushes chips forward':'Taps the felt'):a.action===4?'Both hands push the stack in':'Reaches forward with a raise';
 }
 const t=event.table;
 $('hand').textContent=`HAND ${String(event.hand).padStart(4,'0')}`;
 $('street').textContent=t.street;$('pot').textContent=t.pot;
 $('opp-stack').textContent=t.stacks[1];$('fly-stack').textContent=t.stacks[0];
 $('fly-button').textContent=t.button===0?'Ⓓ':'';$('opp-button').textContent=t.button===1?'Ⓓ':'';
 cards('hole',t.hole,2);cards('opponent-cards',t.opponent_hole,2);cards('board',t.board,5);
 $('history').replaceChildren();
 for(const h of playedHistory.slice(-6)){const span=document.createElement('span');span.className='history-chip';span.textContent=`${h.actor===0?'Fly':otherName} · ${h.name}${h.paid?' '+h.paid:''}`;$('history').append(span)}
 if(!playedHistory.length)$('history').textContent=event.setup_action_count?'River subgame · fixed check/call setup completed; waiting for a neural action.':`Blinds posted. A fresh ${event.stack_bb||20} BB hand.`;
 $('action-callout').textContent=event.kind==='settlement'?`Settled · ${event.net_bb>=0?'+':''}${event.net_bb} BB to fly${event.plasticity_enabled===false?' · weights frozen':''}`:event.kind==='reinforcement'?`Settled · ${event.plasticity.raw_net_bb>=0?'+':''}${event.plasticity.raw_net_bb} BB to fly`:event.action?`${event.actor===0?'Fly':otherName} → ${event.action.name}`:'Information enters the circuit';
 if(event.decision){
  const d=event.decision;decisionHand=event.hand;showActivity(d);
  $('chosen').textContent=names[d.selected];$('decision-context').textContent=`Hand ${event.hand} · decision #${event.sequence} · before action commit`;
  const max=Math.max(...d.scores,.01);
  for(let i=0;i<5;i++){const row=$(`score-${i}`);row.className=`score ${d.legal_mask[i]?'':'illegal'} ${i===d.selected?'selected':''}`;row.querySelector('.mask').textContent=d.legal_mask[i]?'01':'00';row.querySelector('.score-value').textContent=d.scores[i].toFixed(3);row.querySelector('i').style.width=`${100*Math.max(0,d.scores[i])/max}%`}
  $('temperature').textContent=d.temperature.toFixed(2);$('silent').textContent=`${d.silent?'yes':'no'} / ${d.decoder_fallback?'yes':'no'}`;
  $('input-hash').textContent=d.encoded_hash.slice(0,14);$('input-hash').title=d.encoded_hash;
  $('channels').replaceChildren();for(const [label,value] of d.encoded){const line=document.createElement('div');const name=document.createElement('span'),v=document.createElement('span');name.textContent=label;v.textContent=Number(value).toFixed(6);line.append(name,v);$('channels').append(line)}
  $('observation').textContent=JSON.stringify(d.observation,null,2);
  $('event-counter').textContent=`EVENT ${event.sequence} · NEURAL TIME ${(d.virtual_time_ms/1000).toFixed(2)} s`;
 }
 if(event.plasticity){
  const p=event.plasticity;
  const pulseRecord=p.dopamine_pulse;$('recorded-pulse-note').hidden=!pulseRecord;
  if(pulseRecord)$('recorded-pulse-note').textContent=pulseRecord.proxy_population?`Recorded ${pulseRecord.proxy_population} proxy stimulation: ${pulseRecord.pulse_ms} ms; ${pulseRecord.population_spikes.toLocaleString()} population spikes, ${pulseRecord.total_spikes.toLocaleString()} total. Cell-level pulse activity was not recorded.`:'No dopamine pulse was delivered for this recorded update.';
  if(p.activity||p.activity_indices)showActivity(p);else{brainCloud?.setActivity([],[]);spikeTotal=null}
  $('dopamine').replaceChildren(document.createTextNode(p.dopamine===null?'Not delivered':`${p.dopamine>=0?'+':''}${p.dopamine.toFixed(3)}`));if(p.dopamine!==null){const label=document.createElement('span');label.textContent=' RPE';$('dopamine').append(label)}
  const pulse=Math.min(1,Math.abs(p.dopamine))*50;$('pulse').style.width=`${pulse}%`;$('pulse').style.left=`${p.dopamine<0?50-pulse:50}%`;
  $('changed').textContent=p.changed_synapses.toLocaleString();$('weight-delta').textContent=p.absolute_update.toFixed(5);$('eligibility').textContent=p.eligibility_mean===null?'Not used':p.eligibility_mean.toFixed(5);$('ratios').textContent=p.weight_ratio_range.map(x=>x.toFixed(4)).join('–');
  $('event-counter').textContent=p.virtual_time_ms===null?`EVENT ${event.sequence} · RECORDED UPDATE`:`EVENT ${event.sequence} · NEURAL TIME ${(p.virtual_time_ms/1000).toFixed(2)} s`;
 }
 $('spikes').textContent=spikeTotal===null?'UPDATE ACTIVITY · NO CELL-LEVEL RECORD':`${spikeTotal.toLocaleString()} SPIKES · LAST WINDOW`;
 $('return').replaceChildren(document.createTextNode(`${event.return_bb>=0?'+':''}${event.return_bb.toFixed(1)} `));const bb=document.createElement('small');bb.textContent='BB';$('return').append(bb);
 $('event-hash').textContent=event.hash.slice(0,20);$('event-hash').title=event.hash;
 if(event.kind==='opponent_action'&&decisionHand)$('decision-context').textContent=`Last fly decision · hand ${decisionHand} · pre-action information`;
 document.body.dataset.eventKind=event.kind;document.body.dataset.sequence=event.sequence;
}
function connect(){
 socket?.close();socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
 socket.onopen=()=>{if(mode==='live'){$('connection').textContent=trainingServer?'Recorded training · WebSocket':'Live WebSocket';$('lamp').className='online'}};
 socket.onmessage=message=>{if(mode!=='live')return;const event=JSON.parse(message.data);if(!paused)render(event)};
 socket.onerror=()=>{if(mode==='live')$('connection').textContent='Connection error'};
 socket.onclose=()=>{if(mode==='live'){$('connection').textContent='Disconnected';$('lamp').className='';setTimeout(()=>{if(mode==='live')connect()},2000)}};
}
function setMode(value){resetPlasticity();mode=value;$('live').classList.toggle('active',mode==='live');$('replay').classList.toggle('active',mode==='replay');$('play').classList.toggle('active',mode==='play');$('source').textContent=mode==='play'?'YOUR PRIVATE MATCH':mode==='live'?'LIVE STREAM':'CHECKED-IN REPLAY';resetDecision()}
$('live').onclick=async()=>{await stopPlay();clearTimeout(timer);setMode('live');connect()};
async function validateReplay(events){
 let previous='0'.repeat(64);
 // Python canonical JSON is validated by the server for --replay. Browser also
 // checks ordering/link integrity without reserializing floats across languages.
 for(let i=0;i<events.length;i++){if(events[i].sequence!==i||events[i].previous_hash!==previous)throw Error('Broken replay chain');previous=events[i].hash}
}
async function replay(){
 try{
  if(!replayEvents.length){const response=await fetch('/api/example');if(!response.ok)throw Error('Replay unavailable');replayEvents=(await response.text()).trim().split('\n').map(JSON.parse);await validateReplay(replayEvents)}
  setMode('replay');socket?.close();replayIndex=0;$('connection').textContent='Deterministic replay';$('lamp').className='online';clearTimeout(timer);tickReplay();
 }catch(error){errors.push(error.message);$('connection').textContent=error.message}
}
function tickReplay(){if(mode!=='replay')return;if(!paused){render(replayEvents[replayIndex]);replayIndex=(replayIndex+1)%replayEvents.length}timer=setTimeout(tickReplay,Number($('speed').value))}
$('replay').onclick=async()=>{await stopPlay();replay()};
$('pause').onclick=()=>{paused=!paused;avatar?.setPaused(paused);brainCloud?.setPaused(paused);$('pause').textContent=paused?'▶':'Ⅱ';$('pause').setAttribute('aria-label',paused?'Resume display':'Pause display');$('connection').textContent=paused?'Display paused':mode==='live'?(trainingServer?'Recorded training · WebSocket':'Live WebSocket'):'Deterministic replay'};
async function init(){
 const health=await(await fetch('/api/health')).json();trainingServer=Boolean(health.recorded_training);
 if(trainingServer){$('run-label').textContent='RECORDED TRAINING / LOADING VERIFIED EVENTS';$('mode-learning').textContent='Awaiting recorded method';$('mode-teacher').textContent='Awaiting recorded teacher state';$('plasticity-mode').textContent='AWAITING RECORD';$('live').textContent='Training record';$('replay').textContent='Fixture example';$('play').disabled=true;$('play').title='Open a live dashboard to play';sessionStorage.removeItem('flyholdem-play-session')}
 await installGraph(await(await fetch('/api/graph')).json());draw();loadEvidence();
 const saved=sessionStorage.getItem('flyholdem-play-session');if(saved){await resumePlay(saved)}else connect();
 try{const {createFlyViewer}=await import('/assets/fly-avatar.js?v=training-replay-v1');avatar=createFlyViewer($('fly-avatar'),message=>{$('avatar-error').hidden=false;$('avatar-error').textContent=message});if(latest)avatar.update(latest)}
 catch(e){$('avatar-error').hidden=false;$('avatar-error').textContent='3D fly unavailable: '+e.message;errors.push(e.message)}
}
$('view-fly').onclick=()=>{$('avatar-stage').hidden=false;$('table-map').hidden=true;$('view-fly').classList.add('active');$('view-table').classList.remove('active')};
$('view-table').onclick=()=>{$('avatar-stage').hidden=true;$('table-map').hidden=false;$('view-fly').classList.remove('active');$('view-table').classList.add('active')};
$('reset-camera').onclick=()=>avatar?.resetCamera();


function setPlayBusy(value){playBusy=value;$('play').disabled=value||trainingServer;for(const b of $('play-actions').children)b.disabled=value||b.dataset.legal!=='true';$('play-next').disabled=value}
async function playRequest(path,body){
 const response=await fetch(path,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
 const data=await response.json();if(!response.ok)throw Error(data.detail||'Play request failed');return data;
}
function enterPlay(){
 clearTimeout(timer);socket?.close();setMode('play');paused=false;avatar?.setPaused(false);brainCloud?.setPaused(true);brainCloud?.setActivity([],[]);
 $('pause').disabled=true;$('pause').textContent='Ⅱ';$('pause').setAttribute('aria-label','Pause display');
 $('play-controls').hidden=false;$('connection').textContent='Your private match';$('lamp').className='online';
 document.body.dataset.viewer='human';resetDecision();
}
async function stopPlay(){
 ++playEpoch;clearInterval(playHeartbeat);playHeartbeat=null;const token=playSession;playSession=null;sessionStorage.removeItem('flyholdem-play-session');
 $('play-controls').hidden=true;$('pause').disabled=false;document.body.dataset.viewer='spectator';brainCloud?.setPaused(false);setPlayBusy(false);
 if(token){try{await playRequest(`/api/play/${token}/end`,{})}catch(error){$('connection').textContent=error.message}}
}
function playControls(state){
 playRevision=state.revision;$('play-actions').replaceChildren();
 for(const action of state.actions){const button=document.createElement('button');button.textContent=action.name;button.dataset.action=action.index;button.dataset.legal=String(action.legal);button.disabled=!action.legal||playBusy;button.onclick=()=>takePlayAction(action.index);$('play-actions').append(button)}
 $('play-next').hidden=!state.done;$('play-next').disabled=playBusy;
 $('play-status').textContent=state.halted?'Neural player stopped. Start a new match.':state.done?`Hand complete · Your match result ${state.hero_return_bb>=0?'+':''}${state.hero_return_bb.toFixed(1)} BB`:state.human_turn?'Your turn':'The fly is thinking…';
}
async function applyPlay(state,epoch,animate=true){
 if(epoch!==playEpoch||mode!=='play')return;
 playSession=state.session;sessionStorage.setItem('flyholdem-play-session',playSession);
 for(const event of state.events){if(epoch!==playEpoch||mode!=='play')return;render(event);if(animate)await new Promise(resolve=>setTimeout(resolve,event.action?850:350))}
 if(epoch===playEpoch&&mode==='play'){setPlayBusy(false);playControls(state)}
}
function heartbeatPlay(){
 clearInterval(playHeartbeat);playHeartbeat=setInterval(async()=>{if(mode!=='play'||!playSession||playBusy)return;try{await playRequest(`/api/play/${playSession}`)}catch(error){$('play-status').textContent=error.message}},25000);
}
async function startPlay(){
 const epoch=++playEpoch;enterPlay();setPlayBusy(true);$('play-status').textContent='Preparing a frozen neural player…';$('play-actions').replaceChildren();$('play-next').hidden=true;
 try{const state=await playRequest('/api/play/start',{});if(epoch!==playEpoch){playRequest(`/api/play/${state.session}/end`,{}).catch(()=>{});return}await applyPlay(state,epoch);heartbeatPlay()}
 catch(error){if(epoch===playEpoch){setPlayBusy(false);$('play-status').textContent=error.message}}
}
async function resumePlay(token){
 const epoch=++playEpoch;enterPlay();setPlayBusy(true);
 try{const state=await playRequest(`/api/play/${token}`);await applyPlay(state,epoch,false);heartbeatPlay()}
 catch(error){sessionStorage.removeItem('flyholdem-play-session');setPlayBusy(false);$('play-status').textContent='This match has expired. Start a new match.'}
}
async function takePlayAction(action){
 if(playBusy||!playSession)return;const epoch=playEpoch;setPlayBusy(true);$('play-status').textContent='Action submitted · the fly is thinking…';
 try{await applyPlay(await playRequest(`/api/play/${playSession}/action`,{revision:playRevision,action}),epoch)}
 catch(error){if(epoch===playEpoch){setPlayBusy(false);$('play-status').textContent=error.message;try{const state=await playRequest(`/api/play/${playSession}`);playControls(state)}catch{}}}
}
$('play').onclick=startPlay;
$('play-next').onclick=async()=>{if(playBusy||!playSession)return;const epoch=playEpoch;setPlayBusy(true);$('play-status').textContent='Dealing…';try{await applyPlay(await playRequest(`/api/play/${playSession}/hand`,{revision:playRevision}),epoch)}catch(error){setPlayBusy(false);$('play-status').textContent=error.message}};

function showActivity(data){
 if(data.activity){activity=data.activity;spikeTotal=activity.reduce((a,b)=>a+b,0)}
 else{spikeTotal=data.activity_total||0;brainCloud?.setActivity(data.activity_indices||[],data.activity_counts||[])}
}
async function installGraph(value){
 brainCloud?.dispose();brainCloud=null;graph=value;
 const native=Boolean(value.native_cloud);document.body.dataset.nativeGraph=String(native);$('brain').hidden=native;$('brain-native').hidden=!native;$('brain-tools').hidden=!native;
 $('brain-title').textContent=native?(value.mode.includes('shuffled-control')?'Shuffled-connectome control':value.mode.startsWith('fixture')?'Native numerical fixture':value.mode==='full'?'Full retained connectome':'Mushroom-body circuit'):'Fixture circuit';
 $('brain-note').textContent=native?`${value.neuron_count.toLocaleString()} ${value.mode.startsWith('fixture')?'synthetic cells':value.mode.includes('shuffled-control')?'control nodes':'retained neurons'} · ${value.edge_count.toLocaleString()} edges`:'126 synthetic cells · 1,920 existing edges';
 $('brain-layout').textContent=native?`${value.located_neurons.toLocaleString()} annotated soma positions · ${value.unlocated_neurons.toLocaleString()} without coordinates in the separate grid · no edges drawn`:'Schematic layout · 120 sampled edges drawn · No anatomical claim';
 $('brain-native').setAttribute('aria-label',value.mode.startsWith('fixture')?'Synthetic numerical fixture cells in a schematic grid; no anatomical claim':value.mode.includes('shuffled-control')?'Synthetic shuffled-connectome control nodes; no unchanged-wiring claim':'Retained MaleCNS neurons, with annotated soma coordinates and a separate grid for missing coordinates');
 $('brain-filter').options[0].textContent=value.mode.startsWith('fixture')?'All synthetic cells':value.mode.includes('shuffled-control')?'All control nodes':'All retained neurons';
 $('neuron-detail').hidden=true;
 if(native){const {createBrainCloud}=await import('/assets/brain-cloud.js');
  brainCloud=await createBrainCloud($('brain-native'),value,node=>{$('neuron-detail').hidden=false;$('neuron-detail').textContent=`Body ${node.body_id} · ${node.type||node.class||node.superclass||'unclassified'} · ${node.role} · ${node.has_soma_coordinate?'annotated soma position':'no soma coordinate; schematic grid'}`});
  brainCloud.setPaused(paused||mode==='play');brainCloud.setFilter($('brain-filter').value);
  if(latest?.mode===value.mode&&latest.decision)showActivity(latest.decision);
 }else{activity=Array(value.neuron_count).fill(0);shown=Array(value.neuron_count).fill(0)}
}
async function ensureGraph(value){
 if(!['fixture','fixture-native','fixture-native-shuffled-control','circuit','circuit-shuffled-control','full','full-shuffled-control'].includes(value))throw Error('Unknown event graph mode');
 if(graphLoading?.mode===value)return graphLoading.promise;
 const promise=(async()=>{const response=await fetch('/api/graph?mode='+value);if(!response.ok)throw Error('Event graph unavailable');await installGraph(await response.json())})();
 graphLoading={mode:value,promise};try{await promise}finally{if(graphLoading?.promise===promise)graphLoading=null}
}
$('brain-filter').onchange=()=>brainCloud?.setFilter($('brain-filter').value);
async function loadEvidence(){
 try{const response=await fetch('/api/evidence');if(!response.ok)throw Error('Evidence unavailable');const data=await response.json();$('evidence-scope').textContent=data.scope;$('gate-list').replaceChildren();
  for(const gate of data.gates){const item=document.createElement('span'),dot=document.createElement('i'),name=document.createElement('span'),status=document.createElement('b');dot.className=gate.status==='passed'?'passed':'pending';name.textContent=gate.name;status.textContent=gate.status;item.append(dot,name,status);item.title=gate.detail;$('gate-list').append(item)}
  renderEvidenceComparisons($('evidence-comparisons'),data.experiments);
  const link=document.createElement('a');link.href=data.report;link.textContent='Read the registered evidence ↗';link.target='_blank';link.rel='noopener';$('gate-list').append(link);
 }catch(error){$('evidence-scope').textContent='Registered evidence unavailable: '+error.message;errors.push(error.message)}
}

function draw(){
 const canvas=$('brain'),ctx=canvas.getContext('2d'),r=canvas.getBoundingClientRect(),scale=devicePixelRatio||1;
 if(canvas.width!==Math.round(r.width*scale)||canvas.height!==Math.round(r.height*scale)){canvas.width=Math.round(r.width*scale);canvas.height=Math.round(r.height*scale)}
 ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,r.width,r.height);
 if(graph&&!graph.native_cloud){
  const positions=graph.nodes.map(n=>[r.width*.56+n.x*r.width*.38,r.height*.37+n.y*r.height*.43]);
  shown=shown.map((v,i)=>v+(activity[i]-v)*.065);
  for(const [a,b] of graph.sample_edges){const intensity=Math.min(1,(shown[a]+shown[b])/12);ctx.strokeStyle=`rgba(146,181,152,${.025+intensity*.13})`;ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(...positions[a]);ctx.lineTo(...positions[b]);ctx.stroke()}
  graph.nodes.forEach((n,i)=>{const [x,y]=positions[i],intensity=Math.min(1,shown[i]/8),color=i<96?'180,223,199':i<116?'232,182,155':'184,167,211';if(intensity>.1){ctx.beginPath();ctx.fillStyle=`rgba(${color},${intensity*.11})`;ctx.arc(x,y,4+intensity*5,0,Math.PI*2);ctx.fill()}ctx.beginPath();ctx.fillStyle=`rgba(${color},${.2+.8*intensity})`;ctx.arc(x,y,i<96?1.5+intensity:2.4,0,Math.PI*2);ctx.fill()});
 }
 requestAnimationFrame(draw);
}
init().catch(e=>{errors.push(e.message);$('connection').textContent=e.message});
