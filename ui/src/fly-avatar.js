/**
 * Original poker scene and action rig, informed by DOOMFLY's procedural avatar
 * (nftechie/doomfly @ 71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33,
 * doom-ui/lib/fly-model.ts, MIT, (c) 2026 nftechie and DOOMFLY contributors).
 * Its ellipsoid/rod/veined-wing construction ideas are adapted here. Notice:
 * licenses/DOOMFLY-MIT.txt. Poker posing, hands, cards and state mapping are original.
 * This is an illustrative puppet. It cannot choose or submit a poker action.
 */
import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const V=(x,y,z)=>new T.Vector3(x,y,z);
const Y=V(0,1,0);
const palette={shell:0x646958,dark:0x242b23,eye:0xb54530,ivory:0xc8bea0};
const smooth=t=>t*t*(3-2*t);
const ease=t=>smooth(Math.max(0,Math.min(1,t)));
const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);

function material(color,options={}){return new T.MeshStandardMaterial({color,roughness:.66,...options})}
function ball(parent,mat,p,s,detail=2){
 const mesh=new T.Mesh(new T.IcosahedronGeometry(1,detail),mat);
 mesh.position.set(...p);mesh.scale.set(...s);parent.add(mesh);mesh.castShadow=true;return mesh;
}
function segment(parent,a,b,radius,mat){
 const mesh=new T.Mesh(new T.CylinderGeometry(radius*.8,radius,1,7),mat);
 parent.add(mesh);mesh.castShadow=true;placeSegment(mesh,a,b);return mesh;
}
function placeSegment(mesh,a,b){const p=V(...a),q=V(...b);mesh.position.copy(p).add(q).multiplyScalar(.5);mesh.scale.y=p.distanceTo(q);mesh.quaternion.setFromUnitVectors(Y,q.sub(p).normalize())}
function line(parent,points,color,opacity=.4){const mesh=new T.Line(new T.BufferGeometry().setFromPoints(points.map(p=>V(...p))),new T.LineBasicMaterial({color,transparent:true,opacity}));parent.add(mesh);return mesh}

function makeFly(){
 const root=new T.Group(),torso=new T.Group();root.add(torso);
 const shell=material(palette.shell,{metalness:.28,flatShading:true}),dark=material(palette.dark),ivory=material(palette.ivory,{metalness:.3});
 const eye=material(palette.eye,{metalness:.25,roughness:.34,flatShading:true});
 ball(torso,dark,[0,1.04,-1.19],[.42,.58,.46]);
 for(let i=0;i<5;i++){
  const stripe=new T.Mesh(new T.TorusGeometry(.37-i*.025,.035,7,28),i%2?shell:dark);
  stripe.rotation.x=Math.PI/2;stripe.position.set(0,.78+i*.13,-1.19);torso.add(stripe);
 }
 ball(torso,shell,[0,1.65,-1.02],[.47,.56,.45]);
 const head=new T.Group();head.position.set(0,2.28,-.91);torso.add(head);
 ball(head,shell,[0,0,0],[.48,.43,.38]);
 for(const side of [-1,1]){
  const e=ball(head,eye,[side*.33,.03,.17],[.29,.36,.26]);e.rotation.z=-side*.10;
  // Actual geometry facets, distributed deterministically over the compound eye.
  for(let row=-3;row<=3;row++)for(let col=-3;col<=3;col++){
   const r=(row/3.6)**2+(col/3.6)**2;if(r>.85)continue;
   ball(head,eye,[side*.33+row*.069,.03+col*.087,.17+.245*Math.sqrt(1-r)],[.036,.040,.025],0);
  }
  segment(head,[side*.11,.27,.26],[side*.18,.49,.33],.017,dark);
  segment(head,[side*.18,.49,.33],[side*.29,.62,.35],.008,ivory);
  for(let i=0;i<3;i++)segment(head,[side*(.2+i*.025),.50+i*.035,.33],[side*(.30+i*.035),.55+i*.035,.32],.003,dark);
  // Halteres and two posterior pairs of legs: six appendages including the hands.
  segment(torso,[side*.38,1.68,-1.2],[side*.72,1.83,-1.4],.018,dark);
  ball(torso,ivory,[side*.72,1.83,-1.4],[.06,.06,.06],1);
  for(let i=0;i<2;i++){
   const shoulder=[side*.35,1.15+i*.28,-1.1];
   const knee=[side*(.73+i*.14),.45,-1.12+i*.25];
   const toe=[side*(.82+i*.16),.08,-.85+i*.24];
   segment(torso,shoulder,knee,.035,dark);segment(torso,knee,toe,.021,ivory);
   segment(torso,toe,[toe[0]+side*.12,.065,toe[2]+.13],.012,dark);
  }
  for(let i=0;i<10;i++){
   const angle=i*.60;const x=side*(.17+.2*Math.sin(angle)),y=1.65+.45*Math.cos(angle);
   segment(torso,[x,y,-.75],[x+side*.05,y+.09,-.70],.004,dark);
  }
 }
 // Visible proboscis and palps complete the fly's face.
 ball(head,dark,[0,-.22,.36],[.10,.13,.09]);
 segment(head,[0,-.24,.4],[0,-.34,.48],.032,ivory);
 const wings=[];
 for(const side of [-1,1]){
  const pivot=new T.Group();pivot.position.set(side*.26,1.94,-1.22);torso.add(pivot);wings.push(pivot);
  const points=[[0,0],[.24,.5],[.71,1.0],[1.13,1.10],[1.30,.84],[1.10,.41],[.60,.08]];
  const shape=new T.Shape();points.forEach(([x,y],i)=>i?shape.lineTo(x*side,y):shape.moveTo(x*side,y));shape.closePath();
  const geo=new T.ShapeGeometry(shape);const mat=material(0xc6d8b9,{transparent:true,opacity:.28,side:T.DoubleSide,depthWrite:false,roughness:.22,metalness:.2});
  const wing=new T.Mesh(geo,mat);pivot.add(wing);
  for(const end of [[1.12,1.05],[1.22,.81],[1.02,.45],[.60,.13]])line(pivot,[[0,0,.003],[side*end[0],end[1],.003]],0x9fab8e,.52);
  line(pivot,[...points,points[0]].map(([x,y])=>[side*x,y,.005]),0xb5c29f,.55);
  line(pivot,[[side*.24,.5,.004],[side*.67,.52,.004],[side*1.02,.45,.004]],0x9fab8e,.42);
 }
 const arms=[];
 for(const side of [-1,1]){
  const shoulder=[side*.36,1.72,-.75], elbow=[side*.78,1.20,-.60],rest=[side*.43,1.13,-.27];
  const upper=segment(root,shoulder,elbow,.041,dark),fore=segment(root,elbow,rest,.031,ivory);
  const joint=ball(root,shell,elbow,[.07,.07,.07],1);
  const palm=new T.Group();root.add(palm);ball(palm,dark,[0,0,0],[.095,.075,.08]);
  for(let f=-1;f<=1;f++){
   segment(palm,[f*.042,0,.02],[f*.052,-.035,.105],.011,ivory);
   segment(palm,[f*.052,-.035,.105],[f*.043,-.064,.145],.008,dark);
  }
  segment(palm,[side*.07,0,0],[side*.115,-.04,.06],.016,dark);
  arms.push({side,shoulder,elbow,rest,upper,fore,joint,palm});
 }
 return {root,head,torso,wings,arms,pose(left,right,lean,t){
  torso.rotation.x=lean;head.rotation.x=.12;head.rotation.z=Math.sin(t*.72)*.018;head.rotation.y=Math.sin(t*.51)*.035;
  wings.forEach((w,i)=>{w.rotation.y=(i===0?1:-1)*(.27+Math.sin(t*9)*.035);w.rotation.z=(i===0?-1:1)*.09});
  for(let i=0;i<2;i++){
   const a=arms[i],target=i===0?left:right,knee=mix(a.elbow,[target[0]*1.45,target[1]+.10,target[2]-.28],.58);
   const shoulder=[...a.shoulder];shoulder[2]+=lean*.28;
   placeSegment(a.upper,shoulder,knee);placeSegment(a.fore,knee,target);a.joint.position.set(...knee);a.palm.position.set(...target);
   a.palm.rotation.x=target[1]<.97?-.25:-.55;
  }
 }};
}

// A deliberately abstract second player: no human identity or invented neural body.
function makeOpponent(){
 const root=new T.Group(),body=new T.Group();root.add(body);
 const stone=material(0x84948c,{roughness:.85,flatShading:true}),light=material(0xc6cec0,{roughness:.74}),jointMat=material(0x42544a);
 ball(body,stone,[0,1.40,-1.05],[.39,.65,.31],1);
 ball(body,light,[0,2.28,-1.02],[.32,.40,.30],2);
 // A single unbroken seam keeps the head faceless.
 const seam=new T.Mesh(new T.TorusGeometry(.316,.009,5,40),jointMat);
 seam.position.set(0,2.27,-1.02);seam.rotation.x=Math.PI/2;body.add(seam);
 for(const side of [-1,1]){
  segment(body,[side*.18,1.02,-1.07],[side*.34,.48,-1.07],.09,stone);
  segment(body,[side*.34,.48,-1.07],[side*.37,.13,-.94],.065,light);
  ball(body,jointMat,[side*.37,.10,-.86],[.11,.075,.22],1);
 }
 const arms=[-1,1].map(side=>{
  const shoulder=[side*.36,1.72,-.75],elbow=[side*.78,1.20,-.60],rest=[side*.43,1.13,-.27];
  const upper=segment(root,shoulder,elbow,.085,stone),fore=segment(root,elbow,rest,.058,light);
  const joint=ball(root,jointMat,elbow,[.095,.095,.095],1),palm=new T.Group();root.add(palm);
  ball(palm,light,[0,0,.025],[.10,.055,.13],1);
  for(const f of [-1,0,1])segment(palm,[f*.045,0,.075],[f*.048,-.027,.18],.016,light);
  return {shoulder,elbow,upper,fore,joint,palm};
 });
 return {root,pose(left,right,lean,t){
  body.rotation.x=lean;body.rotation.z=Math.sin(t*.55)*.009;
  arms.forEach((a,i)=>{
   const target=i===0?left:right,knee=mix(a.elbow,[target[0]*1.45,target[1]+.10,target[2]-.28],.58);
   const shoulder=[...a.shoulder];shoulder[2]+=lean*.28;
   placeSegment(a.upper,shoulder,knee);placeSegment(a.fore,knee,target);a.joint.position.set(...knee);a.palm.position.set(...target);
   a.palm.rotation.x=target[1]<.97?-.25:-.55;
  });
 }};
}

const textureCache=new Map();
function cardTexture(code){
 if(textureCache.has(code))return textureCache.get(code);
 const canvas=document.createElement('canvas');canvas.width=256;canvas.height=356;
 const ctx=canvas.getContext('2d');ctx.fillStyle='#edead8';ctx.fillRect(0,0,256,356);
 ctx.strokeStyle='#c8c8b6';ctx.lineWidth=6;ctx.strokeRect(7,7,242,342);
 if(code==='??'){
  ctx.fillStyle='#334e3e';ctx.fillRect(14,14,228,328);ctx.strokeStyle='#92b098';ctx.lineWidth=1;
  for(let i=-320;i<320;i+=16){ctx.beginPath();ctx.moveTo(i,14);ctx.lineTo(i+320,342);ctx.stroke()}
  ctx.fillStyle='#c6d2ba';ctx.font='70px Georgia';ctx.textAlign='center';ctx.fillText('◇',128,200);
 }else if(/^[2-9TJQKA][cdhs]$/.test(code)){
  const rank=code[0]==='T'?'10':code[0],suit={c:'♣',d:'♦',h:'♥',s:'♠'}[code[1]];
  ctx.fillStyle='dh'.includes(code[1])?'#a64234':'#20372a';ctx.font='bold 55px Georgia';ctx.fillText(rank,24,65);ctx.font='44px Georgia';ctx.fillText(suit,25,113);
  ctx.textAlign='center';ctx.font='bold 80px Georgia';ctx.fillText(rank,128,209);ctx.font='68px Georgia';ctx.fillText(suit,128,278);
  ctx.save();ctx.translate(256,356);ctx.rotate(Math.PI);ctx.textAlign='left';ctx.font='bold 55px Georgia';ctx.fillText(rank,24,65);ctx.font='44px Georgia';ctx.fillText(suit,25,113);ctx.restore();
 }
 const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;texture.anisotropy=4;textureCache.set(code,texture);return texture;
}
function cardMesh(code,w=.48,h=.67){
 const mesh=new T.Mesh(new T.BoxGeometry(w,h,.014),[material(0xd1d0bd),material(0xd1d0bd),material(0xd1d0bd),material(0xd1d0bd),material(0xffffff,{map:cardTexture(code),roughness:.8}),material(0x334e3e)]);
 mesh.castShadow=true;return mesh;
}
function changeCard(mesh,code){
 const front=mesh.userData.front||mesh.material[4];front.map=cardTexture(code);front.needsUpdate=true;mesh.userData.card=code;
}
function curvedCard(code){
 const group=new T.Group(),geometry=new T.PlaneGeometry(.56,.78,8,20);
 const front=material(0xffffff,{map:cardTexture(code),roughness:.84,side:T.FrontSide});
 const back=material(0xffffff,{map:cardTexture('??'),roughness:.84,side:T.BackSide});
 const face=new T.Mesh(geometry,front),reverse=new T.Mesh(geometry,back);face.castShadow=true;
 group.add(face,reverse);group.userData.front=front;group.userData.geometry=geometry;group.userData.bend=null;
 changeCard(group,code);bendCard(group,.12);return group;
}
function bendCard(card,bend){
 if(Math.abs(card.userData.bend-bend)<.0001)return;
 const geometry=card.userData.geometry,p=geometry.attributes.position;
 for(let i=0;i<p.count;i++){const v=p.getY(i)/.78+.5;p.setZ(i,-bend*v*v)}
 p.needsUpdate=true;geometry.computeVertexNormals();card.userData.bend=bend;
}


const chipCache=new Map();
const chipGeometry=new T.CylinderGeometry(.105,.105,.031,24);
function chip(parent,color,p){
 if(!chipCache.has(color)){
  const top=document.createElement('canvas');top.width=top.height=96;const c=top.getContext('2d');
  const hex='#'+color.toString(16).padStart(6,'0');c.fillStyle='#e0d7b8';c.fillRect(0,0,96,96);c.strokeStyle=hex;c.lineWidth=9;c.beginPath();c.arc(48,48,32,0,Math.PI*2);c.stroke();c.lineWidth=3;c.beginPath();c.arc(48,48,18,0,Math.PI*2);c.stroke();
  const tex=new T.CanvasTexture(top);tex.colorSpace=T.SRGBColorSpace;
  const strip=document.createElement('canvas');strip.width=192;strip.height=16;const sc=strip.getContext('2d');sc.fillStyle=hex;sc.fillRect(0,0,192,16);sc.fillStyle='#e0d7b8';for(let i=0;i<6;i++)sc.fillRect(i*32,0,12,16);
  const side=new T.CanvasTexture(strip);side.colorSpace=T.SRGBColorSpace;
  chipCache.set(color,[material(0xffffff,{map:side}),material(0xffffff,{map:tex}),material(color)]);
 }
 const group=new T.Group();group.position.set(...p);parent.add(group);
 const disk=new T.Mesh(chipGeometry,chipCache.get(color));disk.castShadow=true;group.add(disk);return group;
}
function pile(parent,x,z,color,count=10){const group=new T.Group();parent.add(group);for(let i=0;i<count;i++)chip(group,color,[x,.81+i*.033,z]);return group}

const restLeft=[-.43,1.13,-.27],restRight=[.43,1.13,-.27];
// Each seat consumes its own committed action. Neither rig reads a hidden hand.
function cardOrientation(side){
 const normal=V(side*.78,.52,-.35).normalize(),right=V(0,1,0).cross(normal).normalize(),up=normal.clone().cross(right);
 return new T.Quaternion().setFromRotationMatrix(new T.Matrix4().makeBasis(right,up,normal));
}
function playerRig(model,x,yaw,color){
 const seat=new T.Group();seat.position.set(x,0,0);seat.rotation.y=yaw;seat.add(model.root);
 const cards=[curvedCard('??'),curvedCard('??')];cards.forEach(c=>seat.add(c));
 const holding=cardOrientation(x<0?-1:1),faceDown=new T.Quaternion().setFromEuler(new T.Euler(Math.PI/2,0,0)),faceUp=new T.Quaternion().setFromEuler(new T.Euler(-Math.PI/2,0,0));
 const bank=new T.Group();seat.add(bank);
 // Three readable stacks sit within the rail, mirrored for the second seat.
 for(let i=0;i<18;i++){const stack=i%3,layer=Math.floor(i/3);chip(bank,color,[(x<0?1:-1)*(.63+(stack-1)*.23),.81+layer*.033,.15+(stack===1?.19:0)])}
 const flying=new T.Group();seat.add(flying);
 const movingLeft=pile(flying,-.27,0,color,6),movingRight=pile(flying,.27,0,color,6);flying.visible=false;
 return {seat,model,cards,holding,faceDown,faceUp,bank,flying,movingLeft,movingRight,action:null,elapsed:0,actionHash:null,
  actionFrames:0,left:[...restLeft],right:[...restRight],folded:false,laidDown:false};
}
function resetPlayer(p){p.action=null;p.elapsed=0;p.folded=false;p.laidDown=false;p.actionHash=null;p.actionFrames=0;p.flying.visible=false}
function animatePlayer(p,clock,paused){
 const a=p.action,t=Math.min(1,p.elapsed/.82),reach=Math.sin(Math.PI*t),push=ease(Math.min(1,t*1.4));
 let left=[...restLeft],right=[...restRight],lean=0;p.flying.visible=false;
 if(a){
  if(!paused&&t<1)p.actionFrames++;
  if(a.id===0){left=mix(left,[-.05,.89,.85],push);right=mix(right,[.52,1.03,.20],reach);lean=.05*reach}
  else if(a.check){right=[.45,.91+.18*Math.abs(Math.cos(t*Math.PI*3)),.13];if(t>=1)right=[...restRight]}
  else if(a.id===1){right=mix(right,[.17,.91,.77],reach);lean=.05*reach}
  else if(a.id===2||a.id===3){right=mix(right,[.05,.90,a.id===3?1.02:.77],reach);lean=.08*reach}
  else if(a.id===4){left=mix(left,[-.26,.90,.91],reach);right=mix(right,[.26,.90,.91],reach);lean=.13*reach}
  if(a.paid>0&&t<1){
   p.flying.visible=true;p.flying.position.set(a.id===4?0:.45*(1-push),0,-.20+1.12*push);
   p.movingLeft.visible=a.id===4;p.movingRight.visible=true;
   const n=Math.min(6,Math.max(1,Math.ceil(a.paid/4)));
   for(const stack of [p.movingLeft,p.movingRight])stack.children.forEach((c,i)=>c.visible=i<n);
  }
 }
 p.cards.forEach((c,i)=>{
  const fold=p.folded?ease(Math.min(1,p.elapsed/.75)):0,lay=p.laidDown?ease(Math.min(1,p.elapsed/.20)):0;
  // The printed side points into the holder's information space. Its upper edge
  // curls toward the spectator; the back is a different, one-sided material.
  const x=-.28+(i-.5)*.32;
  c.position.set(x+fold*.30,1.20-Math.max(fold*.408,lay*.408),-.36+fold*1.08+lay*.75);
  c.quaternion.copy(p.holding);
  if(fold)c.quaternion.slerp(p.faceDown,fold);else if(lay)c.quaternion.slerp(p.faceUp,lay);
  c.rotateZ((i===0?1:-1)*.10*(1-lay));
  bendCard(c,.12*(1-Math.max(fold,lay)));
 });
 p.left=left;p.right=right;p.model.pose(left,right,lean,clock);
}

export function createFlyViewer(canvas,onFailure){
 const renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:true,powerPreference:'low-power'});
 renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0x111812,0);
 renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
 renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.35;
 const scene=new T.Scene();scene.add(new T.HemisphereLight(0xe7edd8,0x28312a,2.2));
 const key=new T.DirectionalLight(0xffdfb2,3.7);key.position.set(-3,6,4);key.castShadow=true;key.shadow.mapSize.set(1024,1024);key.shadow.camera.left=-5;key.shadow.camera.right=5;key.shadow.camera.top=5;key.shadow.camera.bottom=-5;key.shadow.bias=-.0005;scene.add(key);
 const rim=new T.DirectionalLight(0xbfdcc7,3.2);rim.position.set(3,3,-4);scene.add(rim);
 const camera=new T.PerspectiveCamera(36,1,.1,50);
 const controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.enablePan=false;controls.minDistance=5;controls.maxDistance=16;
 controls.minPolarAngle=.28;controls.maxPolarAngle=1.40;controls.minAzimuthAngle=-.85;controls.maxAzimuthAngle=.85;
 function resetCamera(){camera.position.set(0,3.42,6.25);controls.target.set(0,1.35,-.12);controls.update()}
 resetCamera();
 const floor=new T.Mesh(new T.CircleGeometry(8,80),material(0x18221b,{roughness:1}));floor.rotation.x=-Math.PI/2;floor.position.y=.015;floor.receiveShadow=true;scene.add(floor);
 const table=new T.Mesh(new T.CylinderGeometry(2.35,2.38,.14,96),material(0x1d3528,{roughness:.95}));table.position.set(0,.70,.22);table.scale.set(1,1,.66);table.receiveShadow=true;scene.add(table);
 const rail=new T.Mesh(new T.TorusGeometry(2.35,.080,10,96),material(0x413e2e,{roughness:.8}));rail.rotation.x=Math.PI/2;rail.scale.set(1,.66,1);rail.position.set(0,.77,.22);scene.add(rail);
 const inner=new T.Mesh(new T.TorusGeometry(2.16,.008,5,96),material(0x728767,{roughness:.9}));inner.rotation.x=Math.PI/2;inner.scale.set(1,.66,1);inner.position.set(0,.776,.22);scene.add(inner);
 const pedestal=new T.Mesh(new T.CylinderGeometry(.28,.53,.61,10),material(0x28352b));pedestal.position.set(0,.35,.22);scene.add(pedestal);
 const potRing=new T.Mesh(new T.TorusGeometry(.30,.005,4,64),material(0x708465));potRing.rotation.x=Math.PI/2;potRing.position.set(0,.782,-.28);scene.add(potRing);
 const fly=playerRig(makeFly(),-1.95,.95,0xa16d4e),other=playerRig(makeOpponent(),1.95,-.95,0x8eab9c);
 // Mirror the abstract player's card fan so both hands face their holder.
 const players=[fly,other];players.forEach(p=>scene.add(p.seat));
 const board=Array.from({length:5},(_,i)=>{const c=cardMesh('??',.46,.65);c.position.set((i-2)*.51,.785,.58);c.rotation.x=-Math.PI/2;c.visible=false;scene.add(c);return c});
 const pot=pile(scene,0,-.28,0xb7b189,12);
 let event=null,clock=0,last=performance.now(),paused=false,frame=0,disposed=false,visible=true,lastEventHash=null;
 function setStack(group,value){group.children.forEach((c,i)=>c.visible=i<Math.min(group.children.length,Math.ceil(value/2)))}
 function update(e){
  if(e.hash===lastEventHash)return;const firstForHand=!event||event.hand!==e.hand||event.viewer!==e.viewer||event.mode!==e.mode;event=e;lastEventHash=e.hash;const t=e.table;
  fly.cards.forEach((c,i)=>{changeCard(c,t.hole[i]||'??');c.visible=Boolean(t.hole[i])});
  other.cards.forEach((c,i)=>{changeCard(c,t.opponent_hole[i]||'??');c.visible=Boolean(t.opponent_hole[i])});
  board.forEach((c,i)=>{c.visible=Boolean(t.board[i]);if(c.visible)changeCard(c,t.board[i])});
  setStack(fly.bank,t.stacks[0]);setStack(other.bank,t.stacks[1]);setStack(pot,t.pot);
  if(e.kind==='hand_start'||firstForHand)players.forEach(resetPlayer);
  if(firstForHand&&t.history.length){
   // A reload may resume from the latest public event, after the action.
   for(const p of players){const a=[...t.history].reverse().find(h=>h.actor===players.indexOf(p));if(!a)continue;
    p.action={id:a.action,paid:a.paid,check:a.action===1&&a.paid===0};p.elapsed=2;p.actionHash=e.hash;p.folded=a.action===0;p.laidDown=a.action===4;
   }
  }
  if((e.kind==='decision'&&e.actor===0&&e.action)||(e.kind==='opponent_action'&&e.actor===1)){
   const p=players[e.actor],id=e.action.action;
   p.action={id,paid:e.action.paid,check:id===1&&e.action.paid===0};
   p.elapsed=0;p.actionHash=e.hash;p.actionFrames=0;p.folded=id===0;if(id===4)p.laidDown=true;
  }
 }
 function draw(now){
  if(disposed)return;const dt=Math.min(.05,(now-last)/1000);last=now;
  if(!paused&&visible){clock+=dt;players.forEach(p=>p.elapsed+=dt);frame++}
  const rect=canvas.getBoundingClientRect();
  if(rect.width>0&&rect.height>0){
   if(canvas.width!==Math.round(rect.width*renderer.getPixelRatio())||canvas.height!==Math.round(rect.height*renderer.getPixelRatio())){
    renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;
    // Fit the same two complete silhouettes on a narrow screen.
    camera.fov=2*Math.atan(Math.tan(36*Math.PI/360)*Math.max(1,2.15/camera.aspect))*180/Math.PI;camera.updateProjectionMatrix();
   }
   players.forEach(p=>animatePlayer(p,clock,paused));
   controls.update();renderer.render(scene,camera);
  }
  requestAnimationFrame(draw);
 }
 function playerSnapshot(p){
  p.seat.updateWorldMatrix(true,true);
  const head=p.seat.localToWorld(V(0,2.28,-.91));
  return {action_hash:p.actionHash,action:p.action?.id??null,check:p.action?.check??false,paid:p.action?.paid??0,
   hole:p.cards.filter(c=>c.visible).map(c=>c.userData.card),left:[...p.left],right:[...p.right],elapsed:p.elapsed,action_frames:p.actionFrames,
   position:p.seat.position.toArray(),yaw:p.seat.rotation.y,visible_chip_count:p.bank.children.filter(c=>c.visible).length,
   cards:p.cards.map(c=>{const position=c.getWorldPosition(V(0,0,0)),normal=V(0,0,1).applyQuaternion(c.getWorldQuaternion(new T.Quaternion()));return {
    bend:c.userData.bend,holder_facing:normal.dot(head.clone().sub(position).normalize()),viewer_facing:normal.dot(camera.position.clone().sub(position).normalize()),
    printed_side:'front-only',back:'pattern',position:position.toArray()};}),
   screen_bounds:(()=>{const b=new T.Box3().setFromObject(p.model.root),points=[];
    for(const x of [b.min.x,b.max.x])for(const y of [b.min.y,b.max.y])for(const z of [b.min.z,b.max.z])points.push(V(x,y,z).project(camera));
    return {left:Math.min(...points.map(v=>v.x)),right:Math.max(...points.map(v=>v.x)),top:Math.max(...points.map(v=>v.y)),bottom:Math.min(...points.map(v=>v.y))};})()
  };
 }
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();onFailure('The 3D view lost its graphics context. Reload to restore the players; the table and neural log remain available.')});
 document.addEventListener('visibilitychange',()=>{visible=!document.hidden;last=performance.now()});requestAnimationFrame(draw);
 return {update,resetCamera,setPaused(value){paused=value},snapshot(){return {
  renderer:'three-webgl2',revision:T.REVISION,event_hash:lastEventHash,...playerSnapshot(fly),opponent:playerSnapshot(other),
  hand:event?.hand??null,board:board.filter(c=>c.visible).map(c=>c.userData.card),paused,frames:frame,
  triangles:renderer.info.render.triangles,draw_calls:renderer.info.render.calls,
  scope:'Illustrated players; gestures follow committed poker actions, not simulated muscles'
 }},dispose(){disposed=true;controls.dispose();renderer.dispose()}};
}
