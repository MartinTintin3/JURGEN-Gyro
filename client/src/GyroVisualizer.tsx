import React from "react";
import { ReactP5Wrapper, Sketch } from "react-p5-wrapper";
import p5 from "p5";

export type GyroVisualizerShaderProps = {
  csvUrl: string;
  width?: number;
  height?: number;
  loopDurationMs?: number;
  palette?: "terrain" | "grayscale";
  smoothingMs?: number;     // EMA time constant
  blendWindowMs?: number;   // linger window around samples
  fieldScale: number;
  warp: number;
  octaves: number;
  speed: number;
  shaderOption: 1 | 2;
  componentConfig: {x:string,y:string,z:string},
};

type Sample = {
  t: number;
  x: number; y: number; z: number;
  seedZ: number;   // noise z
  scale: number;   // spatial freq
  angle: number;   // radians
  hueBias: number; // degrees
  contrast: number;
};

const vertSrc = `
  // vert.glsl
#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

attribute vec3 aPosition;
attribute vec2 aTexCoord;

uniform mat4 uProjectionMatrix;
uniform mat4 uModelViewMatrix;

varying vec2 vUv;

void main() {
  vUv = aTexCoord; // 0..1 across the rect
  gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPosition, 1.0);
}

`;

// Lightweight 3D value noise + smooth interpolation (fast enough for live viz)
// (You can swap in simplex for higher quality if you want.)
const frag1 = `
// frag.glsl — WebGL1-safe (no derivatives), simplex FBM + domain warp
precision mediump float;

varying vec2 vUv;

uniform vec2  u_resolution;
uniform float u_scale;      // ~0.006 from CPU → mapped below
uniform float u_angle;      // radians
uniform float u_seedZ;      // per-timestamp seed (z slice)
uniform float u_contrast;   // lighting contrast
uniform float u_hueBias;    // degrees
uniform int   u_palette;    // 0=terrain(HSB), 1=grayscale

uniform float u_warp;       // e.g. 0.35  (0.2..0.6 nice range)
uniform float u_octaves;    // 1..8 (float; floored to int)

vec2 rot2(vec2 p, float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c)*p; }

vec3 hsb2rgb(float h,float s,float b){
  float hh=mod(h,360.0)/60.0; float c=(b/100.0)*(s/100.0);
  float x=c*(1.0-abs(mod(hh,2.0)-1.0)); vec3 rgb;
  if(hh<1.0) rgb=vec3(c,x,0.0); else if(hh<2.0) rgb=vec3(x,c,0.0);
  else if(hh<3.0) rgb=vec3(0.0,c,x); else if(hh<4.0) rgb=vec3(0.0,x,c);
  else if(hh<5.0) rgb=vec3(x,0.0,c); else rgb=vec3(c,0.0,x);
  float m=(b/100.0)-c; return rgb+vec3(m);
}

/* -------- 3D simplex noise (Ashima/IQ) -------- */
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}

float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=1.0/7.0; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x), p1=vec3(a0.zw,h.y), p2=vec3(a1.xy,h.z), p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m*=m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}

/* -------- FBM + lightweight domain warp -------- */
float fbm(vec3 p, int oct, float lac, float gain){
  float a=0.5, f=1.0, s=0.0;
  for(int i=0;i<8;i++){ if(i>=oct) break; s+=a*snoise(p*f); f*=lac; a*=gain; }
  return 0.5+0.5*s; // 0..1
}

void main(){
  // Aspect correction & rotation
  vec2 aspect = vec2(u_resolution.x/u_resolution.y, 1.0);
  vec2 p = (vUv - 0.5) * aspect;
  p = rot2(p, u_angle);

  // Map small CPU scale to domain frequency
  float F = u_scale * 180.0;

  // Defaults if app doesn't set them
  float warp = (u_warp == 0.0) ? 0.35 : u_warp;
  int   octs = (u_octaves == 0.0) ? 5 : int(floor(u_octaves));

  // Domain-warped position
  vec3 P0 = vec3(p * F, u_seedZ);
  vec2 W = vec2(
    snoise(P0 + vec3(37.2, 11.7, 0.0)),
    snoise(P0 + vec3(-19.3, 44.1, 0.0))
  );
  vec3 Pw = vec3(p * F + warp * W, u_seedZ);

  // Height
  float h = fbm(Pw, octs, 2.0, 0.5);

  // Shading via central differences in **noise space** (no derivatives needed)
  float eps = 1.0;                  // one noise unit
  float hx = fbm(Pw + vec3(eps, 0.0, 0.0), octs, 2.0, 0.5) - h;
  float hy = fbm(Pw + vec3(0.0, eps, 0.0), octs, 2.0, 0.5) - h;
  float shade = clamp(0.7 + (-hx * 1.6 - hy * 1.2) * u_contrast, 0.2, 1.0);

  // Color
  vec3 col;
  if(u_palette == 1){
    float v = clamp(h * shade, 0.0, 1.0);
    col = vec3(v);
  }else{
    float hue = mod(u_hueBias + mix(-40.0, 40.0, h), 360.0);
    float sat = clamp(55.0 + (h - 0.5) * 60.0, 25.0, 90.0);
    float bri = clamp(55.0 + h * 45.0 * shade, 15.0, 100.0);
    col = hsb2rgb(hue, sat, bri);
  }

  gl_FragColor = vec4(col, 1.0);
}


`
const frag2 = `
  // frag.glsl
precision mediump float;

varying vec2 vUv;

uniform vec2  u_resolution;
uniform float u_scale;     // small (~0.006) from CPU
uniform float u_angle;
uniform float u_seedZ;
uniform float u_contrast;
uniform float u_hueBias;   // degrees
uniform int   u_palette;   // 0=terrain(HSB), 1=grayscale

uniform float u_warp;       // ~0.35
uniform float u_octaves;    // integer in [1..8], we floor()

vec2 rot(vec2 p, float a) {
  float c = cos(a), s = sin(a);
  return mat2(c,-s,s,c) * p;
}

// ---- value noise helpers ----
float hash3(vec3 p){
  p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + 3.0));
}
float sCurve(float t){ return t*t*(3.0 - 2.0*t); }

float valueNoise(vec3 p){
  vec3 i = floor(p), f = fract(p);
  vec3 u = vec3(sCurve(f.x), sCurve(f.y), sCurve(f.z));
  float n000 = hash3(i + vec3(0.,0.,0.));
  float n100 = hash3(i + vec3(1.,0.,0.));
  float n010 = hash3(i + vec3(0.,1.,0.));
  float n110 = hash3(i + vec3(1.,1.,0.));
  float n001 = hash3(i + vec3(0.,0.,1.));
  float n101 = hash3(i + vec3(1.,0.,1.));
  float n011 = hash3(i + vec3(0.,1.,1.));
  float n111 = hash3(i + vec3(1.,1.,1.));
  float nx00 = mix(n000, n100, u.x);
  float nx10 = mix(n010, n110, u.x);
  float nx01 = mix(n001, n101, u.x);
  float nx11 = mix(n011, n111, u.x);
  float nxy0 = mix(nx00, nx10, u.y);
  float nxy1 = mix(nx01, nx11, u.y);
  return mix(nxy0, nxy1, u.z);
}

vec3 hsb2rgb(float h, float s, float b){
  float hh = mod(h, 360.0) / 60.0;
  float c = (b / 100.0) * (s / 100.0);
  float x = c * (1.0 - abs(mod(hh, 2.0) - 1.0));
  vec3 rgb;
  if (hh < 1.0) rgb = vec3(c, x, 0.0);
  else if (hh < 2.0) rgb = vec3(x, c, 0.0);
  else if (hh < 3.0) rgb = vec3(0.0, c, x);
  else if (hh < 4.0) rgb = vec3(0.0, x, c);
  else if (hh < 5.0) rgb = vec3(x, 0.0, c);
  else               rgb = vec3(c, 0.0, x);
  float m = (b/100.0) - c;
  return rgb + vec3(m);
}

void main() {
  // UV → centered coords with aspect correction
  vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
  vec2 p = (vUv - 0.5) * aspect;
  p = rot(p, u_angle);

  // IMPORTANT: boost frequency (u_scale is small)
  // Map CPU scale (~0.006) to a useful domain frequency
  float F = u_scale * 180.0;          // try 120–240 if you want
  vec3 P = vec3(p * F, u_seedZ);

  // Height + simple hill-shading (consistent step in noise space)
  float h = valueNoise(P);
  float eps = 1.0;                    // one noise-unit
  float hx = valueNoise(P + vec3(eps, 0.0, 0.0)) - h;
  float hy = valueNoise(P + vec3(0.0, eps, 0.0)) - h;
  float shade = clamp(0.7 + (-hx * 1.6 - hy * 1.2) * u_contrast, 0.2, 1.0);

  vec3 col;
  if (u_palette == 1) {
    float v = clamp(h * shade, 0.0, 1.0);
    col = vec3(v);
  } else {
    float hue = mod(u_hueBias + mix(-40.0, 40.0, h), 360.0);
    float sat = clamp(55.0 + (h - 0.5) * 60.0, 25.0, 90.0);
    float bri = clamp(55.0 + h * 45.0 * shade, 15.0, 100.0);
    col = hsb2rgb(hue, sat, bri);
  }

  gl_FragColor = vec4(col, 1.0);
}

`;

const sketch: Sketch = (p: p5) => {
  // -------- props/state --------
  let W = 900, H = 600;
  let csvUrl = "gyro.csv";
  let loopMs = 12000;
  let palette: "terrain" | "grayscale" = "terrain";
  let smoothingMs = 600;
  let blendWindow = 250;
  let fieldScale = 1;

  let speed = 1;

  let warp = 0.4;
  let octaves = 5;

  // data
  let samples: Sample[] = [];
  let startMs = 0;

  // EMA state
  let ema: Sample | null = null;

  // shader
  let sh1: p5.Shader | null = null;
  let sh2: p5.Shader | null = null;

  let sh: p5.Shader | null = null;

  (p as any).updateWithProps = (props: GyroVisualizerShaderProps) => {
    if (props.width && props.height) {
      if (props.width !== W || props.height !== H) {
        W = props.width; H = props.height;
        if (p.canvas) p.resizeCanvas(W, H);
      }
    }
    if (props.loopDurationMs) loopMs = Math.max(1000, props.loopDurationMs);
    if (props.palette) palette = props.palette;
    if (props.smoothingMs !== undefined) smoothingMs = Math.max(0, props.smoothingMs);
    if (props.blendWindowMs !== undefined) blendWindow = Math.max(0, props.blendWindowMs);
	if (props.fieldScale !== undefined) fieldScale = props.fieldScale;

	if (props.warp != undefined) warp = props.warp;
	if (props.octaves != undefined) octaves = props.octaves;

	if (props.speed !== undefined) speed = props.speed;

	sh = (props.shaderOption == 2 ? sh2 : sh1);

    if (props.csvUrl && props.csvUrl !== csvUrl) {
      csvUrl = props.csvUrl;
      loadCsv();
    }
  };

  p.setup = () => {
    p.createCanvas(W, H, p.WEBGL);        // WEBGL context
    p.pixelDensity(1);                    // predictable u_resolution
    sh1 = p.createShader(vertSrc, frag1);
	sh2 = p.createShader(vertSrc, frag2);
	sh = sh1;
    p.noStroke();
    p.textFont("monospace");
    loadCsv();
  };

  function loadCsv() {
    samples = [];
    ema = null;
    p.noLoop();
    p.loadTable(
      csvUrl, "csv", "header",
      (tbl) => {
        for (let r = 0; r < tbl.getRowCount(); r++) {
          const tsRaw = tbl.getString(r, "timestamp");
          const x = parseFloat(tbl.getString(r, "x"));
          const y = parseFloat(tbl.getString(r, "y"));
          const z = parseFloat(tbl.getString(r, "z"));
          const t = Number.isNaN(Number(tsRaw)) ? Date.parse(tsRaw) : Number(tsRaw);
          if (!Number.isFinite(t)) continue;
          samples.push(makeSample({ t, x, y, z }));
        }
        if (samples.length < 2) { drawError("Need at least 2 rows."); return; }
        samples.sort((a, b) => a.t - b.t);
        startMs = p.millis();
        p.loop();
      },
      () => drawError("Failed to load CSV.")
    );
  }

  function drawError(msg: string) {
    p.background(10);
    p.fill(255);
    p.text(msg, -W * 0.5 + 20, -H * 0.5 + 40);
  }

  // ------- field parameters from a raw row -------
  function makeSample({ t, x, y, z }: { t: number; x: number; y: number; z: number }): Sample {
    const seedZ = hashToUnit(t) * 10.0;
    const scale = 0.006 * (1.0 + 0.8 * sigmoid01(Math.min(5, Math.abs(z) * 0.01)));
    const angle = Math.atan2(y, x) * 0.7;
    const mag = Math.hypot(x, y);
    const hueBias = (p.degrees(angle) + (mag * 0.05)) % 360;
    const contrast = p.constrain(1.0 + Math.sign(z) * Math.min(0.6, Math.abs(z) * 0.002), 0.6, 1.6);
    return { t, x, y, z, seedZ, scale, angle, hueBias, contrast };
  }

  const sigmoid01 = (v: number) => 1 / (1 + Math.exp(-v));
  function hashToUnit(n: number): number {
    let x = Math.imul(n ^ (n >>> 16), 2246822519);
    x = (x ^ (x >>> 13)) * 3266489917;
    x = (x ^ (x >>> 16)) >>> 0;
    return x / 0xffffffff;
  }

  // Quintic smoothstep
  const smoothQuint = (t: number) => t*t*t*(t*(6.0*t - 15.0) + 10.0);

  // Catmull–Rom for scalars
  function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const t2 = t*t, t3 = t2*t;
    return 0.5 * ( (2.0*p1) + (-p0 + p2)*t + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3)*t2 + (-p0 + 3.0*p1 - 3.0*p2 + p3)*t3 );
  }

  function catmullRomAngle(a0: number, a1: number, a2: number, a3: number, t: number): number {
  const unwrap = (a: number, ref: number) => {
    let d = a - ref; // ✅ correct TypeScript
    d = ((d + Math.PI) % (2 * Math.PI)) - Math.PI; // emulate GLSL mod
    return ref + d;
  };

  const p0 = unwrap(a0, a1);
  const p1 = a1;
  const p2 = unwrap(a2, a1);
  const p3 = unwrap(a3, a2);

  const t2 = t * t;
  const t3 = t2 * t;

  return (
    0.5 *
    (2 * p1 +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
  );
}

  function findBracket(tAbs: number): number {
    let i = 0;
    while (i < samples.length - 2 && samples[i + 1].t <= tAbs) i++;
    return i;
  }

  function emaBlend(prev: Sample | null, cur: Sample, a: number): Sample {
    if (!prev) return { ...cur };
    const lerpAngle = (A: number, B: number, t: number) => {
      let d = B - A;
      d = ((d + Math.PI) % (2*Math.PI)) - Math.PI;
      return A + d * t;
    };
    return {
      t: p.lerp(prev.t, cur.t, a),
      x: p.lerp(prev.x, cur.x, a),
      y: p.lerp(prev.y, cur.y, a),
      z: p.lerp(prev.z, cur.z, a),
      seedZ: p.lerp(prev.seedZ, cur.seedZ, a),
      scale: p.lerp(prev.scale, cur.scale, a),
      angle: lerpAngle(prev.angle, cur.angle, a),
      hueBias: p.lerp(prev.hueBias, cur.hueBias, a),
      contrast: p.lerp(prev.contrast, cur.contrast, a),
    };
  }

  p.draw = () => {
    if (!sh || samples.length < 2) return;

    // uniform progress
    const elapsed = (p.millis() - startMs) % loopMs;
    const prog = elapsed / loopMs;

    const tStart = samples[0].t, tEnd = samples[samples.length - 1].t;
    const tAbs = p.lerp(tStart, tEnd, prog);

    const i = findBracket(tAbs);
    const sA = samples[i], sB = samples[i + 1];
    const tA = sA.t - blendWindow, tB = sB.t + blendWindow;
    const uClamped = p.constrain((tAbs - tA) / Math.max(1, tB - tA), 0, 1);
    const u = smoothQuint(uClamped);

    const sPrev = samples[Math.max(0, i - 1)];
    const sNext = samples[Math.min(samples.length - 1, i + 2)];

    const spl: Sample = {
      t: p.lerp(sA.t, sB.t, u),
      x: catmullRom(sPrev.x, sA.x, sB.x, sNext.x, u),
      y: catmullRom(sPrev.y, sA.y, sB.y, sNext.y, u),
      z: catmullRom(sPrev.z, sA.z, sB.z, sNext.z, u),
      seedZ: catmullRom(sPrev.seedZ, sA.seedZ, sB.seedZ, sNext.seedZ, u),
      scale: catmullRom(sPrev.scale, sA.scale, sB.scale, sNext.scale, u),
      angle: catmullRomAngle(sPrev.angle, sA.angle, sB.angle, sNext.angle, u),
      hueBias: catmullRom(sPrev.hueBias, sA.hueBias, sB.hueBias, sNext.hueBias, u),
      contrast: catmullRom(sPrev.contrast, sA.contrast, sB.contrast, sNext.contrast, u),
    };

    const alpha = smoothingMs > 0 ? 1 - Math.exp(-p.deltaTime / smoothingMs) : 1;
    const S = emaBlend(ema, spl, alpha);
    ema = S;

    // Feed uniforms & draw a full-screen quad
    p.shader(sh);
    sh.setUniform("u_resolution", [W, H]);
    sh.setUniform("u_scale", S.scale * fieldScale);
    sh.setUniform("u_angle", S.angle);
    sh.setUniform("u_seedZ", S.seedZ * speed);
    sh.setUniform("u_contrast", S.contrast);
    sh.setUniform("u_hueBias", ((S.hueBias % 360) + 360) % 360);
    sh.setUniform("u_palette", palette === "grayscale" ? 1 : 0);

	sh.setUniform("u_warp", warp);    // 0.25..0.6 nice range
	sh.setUniform("u_octaves", octaves); // float; shader floors to int


    // Draw a single rectangle covering the viewport (WEBGL mode origin is center)
    p.rectMode(p.CENTER);
    p.rect(0, 0, W, H);
  };
};

export default function GyroVisualizer(props: GyroVisualizerShaderProps) {
  const {
    csvUrl,
    width = 1024,
    height = 640,
    loopDurationMs = 12000,
    palette = "terrain",
    smoothingMs = 600,
    blendWindowMs = 250,
	fieldScale = 1,
	warp = 0.4,
	octaves = 5,
	shaderOption = 1,
	speed = 1,
	componentConfig = {x:"x",y:"y",z:"z"},
  } = props;

  return (
    <ReactP5Wrapper
      sketch={sketch}
      csvUrl={csvUrl}
      width={width}
      height={height}
      loopDurationMs={loopDurationMs}
      palette={palette}
      smoothingMs={smoothingMs}
      blendWindowMs={blendWindowMs}
	  fieldScale={fieldScale}
	  warp={warp}
	  octaves={octaves}
	  shaderOption={shaderOption}
	  speed={speed}
	  componentConfig={componentConfig}
    />
  );
}
