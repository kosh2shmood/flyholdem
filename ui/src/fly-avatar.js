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
   const knee=[side*(.83+i*.17),.90,-.99+i*.33];
   const toe=[side*(.95+i*.18),.80,-.40+i*.45];
   segment(torso,shoulder,knee,.035,dark);segment(torso,knee,toe,.021,ivory);
   segment(torso,toe,[toe[0]+side*.12,.80,toe[2]+.13],.012,dark);
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
  torso.rotation.x=lean;head.rotation.z=Math.sin(t*.72)*.018;head.rotation.y=Math.sin(t*.51)*.035;
  wings.forEach((w,i)=>{w.rotation.y=(i===0?1:-1)*(.27+Math.sin(t*9)*.035);w.rotation.z=(i===0?-1:1)*.09});
  for(let i=0;i<2;i++){
   const a=arms[i],target=i===0?left:right,knee=mix(a.elbow,[target[0]*1.45,target[1]+.10,target[2]-.28],.58);
   const shoulder=[...a.shoulder];shoulder[2]+=lean*.28;
   placeSegment(a.upper,shoulder,knee);placeSegment(a.fore,knee,target);a.joint.position.set(...knee);a.palm.position.set(...target);
   a.palm.rotation.x=target[1]<.97?-.25:-.55;
  }
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
  ctx.textAlign='center';ctx.font='116px Georgia';ctx.fillText(suit,128,230);
  ctx.save();ctx.translate(256,356);ctx.rotate(Math.PI);ctx.textAlign='left';ctx.font='bold 55px Georgia';ctx.fillText(rank,24,65);ctx.font='44px Georgia';ctx.fillText(suit,25,113);ctx.restore();
 }
 const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;texture.anisotropy=4;textureCache.set(code,texture);return texture;
}
function cardMesh(code,w=.48,h=.67){
 const mesh=new T.Mesh(new T.BoxGeometry(w,h,.014),[material(0xd1d0bd),material(0xd1d0bd),material(0xd1d0bd),material(0xd1d0bd),material(0xffffff,{map:cardTexture(code),roughness:.8}),material(0x334e3e)]);
 mesh.castShadow=true;return mesh;
}
function changeCard(mesh,code){mesh.material[4].map=cardTexture(code);mesh.material[4].needsUpdate=true;mesh.userData.card=code}

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

export function createFlyViewer(canvas,onFailure){
 const renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:true,powerPreference:'low-power'});
 renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0x111812,0);
 renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
 renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.35;
 const scene=new T.Scene();scene.add(new T.HemisphereLight(0xe7edd8,0x28312a,2.2));
 const key=new T.DirectionalLight(0xffdfb2,3.7);key.position.set(-3,6,4);key.castShadow=true;key.shadow.mapSize.set(1024,1024);key.shadow.camera.left=-5;key.shadow.camera.right=5;key.shadow.camera.top=5;key.shadow.camera.bottom=-5;key.shadow.bias=-.0005;scene.add(key);
 const rim=new T.DirectionalLight(0xbfdcc7,3.2);rim.position.set(3,3,-4);scene.add(rim);
 const camera=new T.PerspectiveCamera(36,1,.1,50);
 const controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.enablePan=false;controls.minDistance=4.2;controls.maxDistance=10;
 controls.minPolarAngle=.28;controls.maxPolarAngle=1.34;controls.minAzimuthAngle=-.95;controls.maxAzimuthAngle=.95;
 function resetCamera(){camera.position.set(0,3.55,6.3);controls.target.set(0,1.35,-.28);controls.update()}
 resetCamera();
 const table=new T.Mesh(new T.CylinderGeometry(3.35,3.40,.18,96),material(0x1d3528,{roughness:.95}));table.position.set(0,.70,.48);table.scale.set(1.22,1,.78);table.receiveShadow=true;scene.add(table);
 const rail=new T.Mesh(new T.TorusGeometry(3.35,.095,10,96),material(0x413e2e,{roughness:.8}));rail.rotation.x=Math.PI/2;rail.scale.set(1.22,.78,1);rail.position.set(0,.79,.48);scene.add(rail);
 const inner=new T.Mesh(new T.TorusGeometry(3.12,.01,5,96),material(0x728767,{roughness:.9}));inner.rotation.x=Math.PI/2;inner.scale.set(1.22,.78,1);inner.position.set(0,.799,.48);scene.add(inner);
 // A ring around the pot marks the same public game area, without invented chips.
 const potRing=new T.Mesh(new T.TorusGeometry(.42,.005,4,64),material(0x708465));potRing.rotation.x=Math.PI/2;potRing.position.set(0,.802,.50);scene.add(potRing);
 const fly=makeFly();scene.add(fly.root);
 const holes=[cardMesh('??'),cardMesh('??')];holes.forEach((c,i)=>{c.position.set((i-.5)*.36,1.18,-.30);c.rotation.set(-.18,0,(i===0?1:-1)*.12);scene.add(c)});
 const board=Array.from({length:5},(_,i)=>{const c=cardMesh('??',.45,.63);c.position.set((i-2)*.53,.808,1.29);c.rotation.x=-Math.PI/2;c.visible=false;scene.add(c);return c});
 const otherCards=[cardMesh('??',.40,.56),cardMesh('??',.40,.56)];otherCards.forEach((c,i)=>{c.position.set(2.4+(i-.5)*.30,.82,-.25);c.rotation.set(-Math.PI/2,0,-.15+i*.3);scene.add(c)});
 const chipRoot=new T.Group();scene.add(chipRoot);
 const banks=[pile(chipRoot,-1.28,-.25,0xa16d4e,12),pile(chipRoot,1.28,-.25,0xb7b189,12)];
 const pot=pile(chipRoot,0,.50,0xa16d4e,12),opponent=pile(chipRoot,2.60,.55,0x81947e,12);
 const flying=new T.Group();scene.add(flying);
 const movingLeft=pile(flying,-.30,0,0xa16d4e,6),movingRight=pile(flying,.30,0,0xb7b189,6);flying.visible=false;
 let event=null,action=null,elapsed=0,clock=0,last=performance.now(),paused=false,frame=0,disposed=false;
 let left=[-.43,1.13,-.27],right=[.43,1.13,-.27],folded=false,visible=true;
 let lastEventHash=null,lastActionHash=null,frameForAction=0;
 function setStack(group,value){group.children.forEach((c,i)=>c.visible=i<Math.min(12,Math.ceil(value/2)))}
 function update(e){
  if(e.hash===lastEventHash)return;
  event=e;lastEventHash=e.hash;
  const t=e.table;
  holes.forEach((c,i)=>{changeCard(c,t.hole[i]||'??');c.visible=Boolean(t.hole[i])});
  board.forEach((c,i)=>{c.visible=Boolean(t.board[i]);if(c.visible)changeCard(c,t.board[i])});
  otherCards.forEach((c,i)=>changeCard(c,t.opponent_hole[i]||'??'));
  setStack(banks[0],t.stacks[0]/2);setStack(banks[1],t.stacks[0]/2);setStack(pot,t.pot);setStack(opponent,t.stacks[1]);
  if(e.kind==='hand_start'){action=null;elapsed=0;folded=false;flying.visible=false;holes.forEach(c=>c.visible=true)}
  if(e.kind==='decision'&&e.actor===0&&e.decision){
   action={id:e.decision.selected,paid:e.action.paid,check:e.decision.selected===1&&e.decision.observation.to_call===0};
   elapsed=0;lastActionHash=e.hash;frameForAction=0;folded=action.id===0;
  }
 }
 function draw(now){
  if(disposed)return;
  const dt=Math.min(.05,(now-last)/1000);last=now;
  if(!paused&&visible){clock+=dt;elapsed+=dt;frame++}
  const rect=canvas.getBoundingClientRect();
  if(rect.width>0&&rect.height>0){
   if(canvas.width!==Math.round(rect.width*renderer.getPixelRatio())||canvas.height!==Math.round(rect.height*renderer.getPixelRatio())){
    renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();
   }
   left=[-.43,1.13,-.27];right=[.43,1.13,-.27];let lean=0;
   const t=Math.min(1,elapsed/.82),reach=Math.sin(Math.PI*t),push=ease(Math.min(1,t*1.4));
   flying.visible=false;
   if(action){
    if(!paused&&t<1)frameForAction++;
    if(action.id===0){left=mix(left,[-.05,.89,.85],push);right=mix(right,[.52,1.03,.20],reach);lean=.05*reach}
    else if(action.check){right=[.45,.91+.18*Math.abs(Math.cos(t*Math.PI*3)),.13];if(t>=1)right=[.43,1.13,-.27]}
    else if(action.id===1){right=mix(right,[.17,.91,.77],reach);lean=.05*reach}
    else if(action.id===2||action.id===3){right=mix(right,[.05,.90,action.id===3?1.02:.77],reach);left=mix(left,[-.73,.94,-.05],reach);lean=.08*reach}
    else if(action.id===4){left=mix(left,[-.26,.90,.91],reach);right=mix(right,[.26,.90,.91],reach);lean=.13*reach}
    if(action.paid>0&&t<1){
     flying.visible=true;flying.position.set(action.id===4?0:1.0*(1-push),0,-.15+1.12*push);
     movingLeft.visible=action.id===4;movingRight.visible=true;
     const n=Math.min(6,Math.max(1,Math.ceil(action.paid/4)));
     movingRight.children.forEach((c,i)=>c.visible=i<n);movingLeft.children.forEach((c,i)=>c.visible=i<n);
    }
   }
   holes.forEach((c,i)=>{
    const fold=folded?ease(Math.min(1,elapsed/.75)):0;
    c.position.set((i-.5)*.36+fold*.10,1.18-fold*.36,-.30+fold*1.08);
    c.rotation.set(-.18-fold*(Math.PI/2-.18),0,(i===0?1:-1)*(.12+fold*.20));
   });
   fly.pose(left,right,lean,clock);
   controls.update();renderer.render(scene,camera);
  }
  requestAnimationFrame(draw);
 }
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();onFailure('The 3D view lost its graphics context. Reload to restore the fly; the table and neural log remain available.')});
 document.addEventListener('visibilitychange',()=>{visible=!document.hidden;last=performance.now()});
 requestAnimationFrame(draw);
 return {update,resetCamera,setPaused(value){paused=value},snapshot(){return {
  renderer:'three-webgl2',revision:T.REVISION,event_hash:lastEventHash,action_hash:lastActionHash,
  action:action?.id??null,check:action?.check??false,paid:action?.paid??0,hand:event?.hand??null,
  hole:holes.map(c=>c.userData.card),board:board.filter(c=>c.visible).map(c=>c.userData.card),
  left:[...left],right:[...right],elapsed,paused,frames:frame,action_frames:frameForAction,
  triangles:renderer.info.render.triangles,draw_calls:renderer.info.render.calls,
  scope:'Illustrated avatar; gestures map recorded neural actions, not simulated muscles'
 }},dispose(){disposed=true;controls.dispose();renderer.dispose()}};
}
