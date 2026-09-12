const $ = id => document.getElementById(id);
const names = ['Fold', 'Check / call', 'Raise ½ pot', 'Raise pot', 'All-in'];
const suits = {c:'♣', d:'♦', h:'♥', s:'♠'};
let socket, mode='live', paused=false, latest=null, timer, replayEvents=[], replayIndex=0, graph=null;
let activity=Array(126).fill(0), shown=Array(126).fill(0);
let decisionHand=null;
const errors=[];
window.addEventListener('error', e => errors.push(e.message));
window.flyholdem = {get latest(){return latest}, get mode(){return mode}, errors};
function card(value){
 const el=document.createElement('span'); el.className='card';
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
function render(event){
 latest=event;
 if(event.kind==='hand_start')resetDecision();
 const t=event.table;
 $('hand').textContent=`HAND ${String(event.hand).padStart(4,'0')}`;
 $('street').textContent=t.street;$('pot').textContent=t.pot;
 $('opp-stack').textContent=t.stacks[1];$('fly-stack').textContent=t.stacks[0];
 $('fly-button').textContent=t.button===0?'Ⓓ':'';$('opp-button').textContent=t.button===1?'Ⓓ':'';
 cards('hole',t.hole,2);cards('opponent-cards',t.opponent_hole,2);cards('board',t.board,5);
 $('history').replaceChildren();
 for(const h of t.history.slice(-6)){const span=document.createElement('span');span.className='history-chip';span.textContent=`${h.actor===0?'Fly':'Station'} · ${h.name}${h.paid?' '+h.paid:''}`;$('history').append(span)}
 if(!t.history.length)$('history').textContent='Blinds posted. A fresh 20 BB hand.';
 $('action-callout').textContent=event.kind==='reinforcement'?`Settled · ${event.plasticity.raw_net_bb>=0?'+':''}${event.plasticity.raw_net_bb} BB to fly`:event.action?`${event.actor===0?'Fly':'Station'} → ${event.action.name}`:'Information enters the circuit';
 if(event.decision){
  const d=event.decision;decisionHand=event.hand;activity=d.activity;
  $('chosen').textContent=names[d.selected];$('decision-context').textContent=`Hand ${event.hand} · decision #${event.sequence} · before action commit`;
  const max=Math.max(...d.scores,.01);
  for(let i=0;i<5;i++){const row=$(`score-${i}`);row.className=`score ${d.legal_mask[i]?'':'illegal'} ${i===d.selected?'selected':''}`;row.querySelector('.mask').textContent=d.legal_mask[i]?'01':'00';row.querySelector('.score-value').textContent=d.scores[i].toFixed(3);row.querySelector('i').style.width=`${100*d.scores[i]/max}%`}
  $('temperature').textContent=d.temperature.toFixed(2);$('silent').textContent=`${d.silent?'yes':'no'} / ${d.decoder_fallback?'yes':'no'}`;
  $('input-hash').textContent=d.encoded_hash.slice(0,14);$('input-hash').title=d.encoded_hash;
  $('channels').replaceChildren();for(const [label,value] of d.encoded){const line=document.createElement('div');const name=document.createElement('span'),v=document.createElement('span');name.textContent=label;v.textContent=Number(value).toFixed(6);line.append(name,v);$('channels').append(line)}
  $('observation').textContent=JSON.stringify(d.observation,null,2);
  $('event-counter').textContent=`EVENT ${event.sequence} · NEURAL TIME ${(d.virtual_time_ms/1000).toFixed(2)} s`;
 }
 if(event.plasticity){
  const p=event.plasticity;activity=p.activity;
  $('dopamine').replaceChildren(document.createTextNode(`${p.dopamine>=0?'+':''}${p.dopamine.toFixed(3)}`));const label=document.createElement('span');label.textContent=' RPE';$('dopamine').append(label);
  const pulse=Math.min(1,Math.abs(p.dopamine))*50;$('pulse').style.width=`${pulse}%`;$('pulse').style.left=`${p.dopamine<0?50-pulse:50}%`;
  $('changed').textContent=p.changed_synapses.toLocaleString();$('weight-delta').textContent=p.absolute_update.toFixed(5);$('eligibility').textContent=p.eligibility_mean.toFixed(5);$('ratios').textContent=p.weight_ratio_range.map(x=>x.toFixed(4)).join('–');
  $('event-counter').textContent=`EVENT ${event.sequence} · NEURAL TIME ${(p.virtual_time_ms/1000).toFixed(2)} s`;
 }
 $('spikes').textContent=`${activity.reduce((a,b)=>a+b,0).toLocaleString()} SPIKES · LAST WINDOW`;
 $('return').replaceChildren(document.createTextNode(`${event.return_bb>=0?'+':''}${event.return_bb.toFixed(1)} `));const bb=document.createElement('small');bb.textContent='BB';$('return').append(bb);
 $('event-hash').textContent=event.hash.slice(0,20);$('event-hash').title=event.hash;
 if(event.kind==='opponent_action'&&decisionHand)$('decision-context').textContent=`Last fly decision · hand ${decisionHand} · pre-action information`;
 document.body.dataset.eventKind=event.kind;document.body.dataset.sequence=event.sequence;
}
function connect(){
 socket?.close();socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
 socket.onopen=()=>{if(mode==='live'){$('connection').textContent='Live WebSocket';$('lamp').className='online'}};
 socket.onmessage=message=>{if(mode!=='live')return;const event=JSON.parse(message.data);if(!paused)render(event)};
 socket.onerror=()=>{if(mode==='live')$('connection').textContent='Connection error'};
 socket.onclose=()=>{if(mode==='live'){$('connection').textContent='Disconnected';$('lamp').className='';setTimeout(()=>{if(mode==='live')connect()},2000)}};
}
function setMode(value){mode=value;$('live').classList.toggle('active',mode==='live');$('replay').classList.toggle('active',mode==='replay');$('source').textContent=mode==='live'?'LIVE STREAM':'CHECKED-IN REPLAY';resetDecision()}
$('live').onclick=()=>{clearTimeout(timer);setMode('live');connect()};
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
$('replay').onclick=replay;
$('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'▶':'Ⅱ';$('pause').setAttribute('aria-label',paused?'Resume display':'Pause display');$('connection').textContent=paused?'Display paused':mode==='live'?'Live WebSocket':'Deterministic replay'};
async function init(){graph=await(await fetch('/api/graph')).json();connect();draw()}
function draw(){
 const canvas=$('brain'),ctx=canvas.getContext('2d'),r=canvas.getBoundingClientRect(),scale=devicePixelRatio||1;
 if(canvas.width!==Math.round(r.width*scale)||canvas.height!==Math.round(r.height*scale)){canvas.width=Math.round(r.width*scale);canvas.height=Math.round(r.height*scale)}
 ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,r.width,r.height);
 if(graph){
  const positions=graph.nodes.map(n=>[r.width*.56+n.x*r.width*.38,r.height*.37+n.y*r.height*.43]);
  shown=shown.map((v,i)=>v+(activity[i]-v)*.065);
  for(const [a,b] of graph.sample_edges){const intensity=Math.min(1,(shown[a]+shown[b])/12);ctx.strokeStyle=`rgba(146,181,152,${.025+intensity*.13})`;ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(...positions[a]);ctx.lineTo(...positions[b]);ctx.stroke()}
  graph.nodes.forEach((n,i)=>{const [x,y]=positions[i],intensity=Math.min(1,shown[i]/8),color=i<96?'180,223,199':i<116?'232,182,155':'184,167,211';if(intensity>.1){ctx.beginPath();ctx.fillStyle=`rgba(${color},${intensity*.11})`;ctx.arc(x,y,4+intensity*5,0,Math.PI*2);ctx.fill()}ctx.beginPath();ctx.fillStyle=`rgba(${color},${.2+.8*intensity})`;ctx.arc(x,y,i<96?1.5+intensity:2.4,0,Math.PI*2);ctx.fill()});
 }
 requestAnimationFrame(draw);
}
init().catch(e=>{errors.push(e.message);$('connection').textContent=e.message});
