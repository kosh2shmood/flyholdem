import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

export async function createBrainCloud(canvas, graph, onSelect){
 const [positionsResponse,rolesResponse]=await Promise.all([fetch('/api/graph/positions'),fetch('/api/graph/roles')]);
 if(!positionsResponse.ok||!rolesResponse.ok)throw Error('Native neuron geometry unavailable');
 const positions=new Float32Array(await positionsResponse.arrayBuffer()),roles=new Uint8Array(await rolesResponse.arrayBuffer());
 if(positions.length!==graph.neuron_count*3||roles.length!==graph.neuron_count)throw Error('Native point-cloud shape mismatch');
 const renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.01,100);
 camera.position.set(.8,.1,4.2);
 const controls=new OrbitControls(camera,canvas);controls.target.set(.55,0,0);controls.enableDamping=true;
 controls.minDistance=.25;controls.maxDistance=12;controls.update();
 const geometry=new THREE.BufferGeometry();
 geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
 const palette=['#8b9ba5','#b4dfc7','#e9928f','#f6df9b','#9fc6ec','#d6a5e8','#f2b996','#a994da','#e8b69b'].map(c=>new THREE.Color(c));
 const colors=new Float32Array(graph.neuron_count*3),activity=new Float32Array(graph.neuron_count);
 for(let i=0;i<roles.length;i++)palette[roles[i]].toArray(colors,i*3);
 geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
 geometry.setAttribute('activity',new THREE.BufferAttribute(activity,1));
 geometry.setAttribute('role',new THREE.BufferAttribute(new Float32Array(roles),1));
 const material=new THREE.ShaderMaterial({transparent:true,depthWrite:false,vertexColors:true,
  uniforms:{elapsed:{value:0},visibleFilter:{value:0},pixelRatio:{value:renderer.getPixelRatio()}},
  vertexShader:`attribute float activity; attribute float role; varying vec3 vColor; varying float vAlpha;
   uniform float elapsed; uniform float visibleFilter; uniform float pixelRatio;
   void main(){float glow=min(1.0,log(1.0+activity)/4.0)*exp(-elapsed*.7);
    vColor=mix(color,vec3(1.0),glow*.5);vAlpha=.3+glow*.65;
    if(role>0.5)vAlpha+=.15;
    if(visibleFilter>1.5 && activity<.5)vAlpha=0.0;
    else if(visibleFilter>.5 && visibleFilter<1.5 && role<.5)vAlpha=0.0;
    vec4 mv=modelViewMatrix*vec4(position,1.0);gl_Position=projectionMatrix*mv;
    gl_PointSize=clamp((1.3+glow*3.0+(role>1.5&&role<6.5?2.0:0.0))*pixelRatio*4.0/max(.6,-mv.z),1.0,12.0);}`,
  fragmentShader:`varying vec3 vColor;varying float vAlpha;void main(){float d=length(gl_PointCoord-.5);if(d>.5||vAlpha<.01)discard;gl_FragColor=vec4(vColor,vAlpha*smoothstep(.5,.15,d));}`});
 const cloud=new THREE.Points(geometry,material);scene.add(cloud);
 let disposed=false,paused=false,lastEvent=performance.now(),pausedElapsed=0,frame;
 function draw(){if(disposed)return;const rect=canvas.getBoundingClientRect();
  if(rect.width&&rect.height){const w=Math.round(rect.width),h=Math.round(rect.height);if(canvas.width!==Math.round(w*renderer.getPixelRatio())||canvas.height!==Math.round(h*renderer.getPixelRatio())){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}
   material.uniforms.elapsed.value=paused?pausedElapsed:(performance.now()-lastEvent)/1000;
   controls.update();renderer.render(scene,camera)}frame=requestAnimationFrame(draw)}draw();
 const ray=new THREE.Raycaster();ray.params.Points.threshold=.012;let start;
 const down=e=>{start=[e.clientX,e.clientY]};
 const up=async e=>{if(!start||Math.hypot(e.clientX-start[0],e.clientY-start[1])>4)return;
  const rect=canvas.getBoundingClientRect();ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);
  const selection=material.uniforms.visibleFilter.value;const hit=ray.intersectObject(cloud).find(h=>selection>1.5?activity[h.index]>0:selection>.5?roles[h.index]>0:true);if(!hit)return;
  const response=await fetch(`/api/graph/neuron/${hit.index}`);if(response.ok)onSelect(await response.json())};
 canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointerup',up);
 return {setActivity(indices,counts){activity.fill(0);for(let i=0;i<indices.length;i++)activity[indices[i]]=counts[i];geometry.attributes.activity.needsUpdate=true;lastEvent=performance.now();pausedElapsed=0},
  setFilter(value){material.uniforms.visibleFilter.value=Number(value)},setPaused(value){if(value&&!paused)pausedElapsed=(performance.now()-lastEvent)/1000;if(!value&&paused)lastEvent=performance.now()-pausedElapsed*1000;paused=value},
  get neuronCount(){return roles.length},snapshot(){return {neurons:roles.length,drawn_points:renderer.info.render.points,filter:material.uniforms.visibleFilter.value,activity_total:activity.reduce((a,b)=>a+b,0),paused}},dispose(){disposed=true;cancelAnimationFrame(frame);canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointerup',up);controls.dispose();geometry.dispose();material.dispose();renderer.dispose()}};
}
