"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[1550],{1550:(t,e,i)=>{i.r(e),i.d(e,{HLSProvider:()=>o});var s=i(7962),n=i(1129);i(758);class r{#t;#e;#i=null;#s=null;config={};#n=new Set;get instance(){return this.#i}constructor(t,e){this.#t=t,this.#e=e}setup(t){const{streamType:e}=this.#e.$state,i=(0,s.peek)(e).includes("live"),r=(0,s.peek)(e).includes("ll-");this.#i=new t({lowLatencyMode:r,backBufferLength:r?4:i?8:void 0,renderTextTracksNatively:!1,...this.config});const a=this.#r.bind(this);for(const s of Object.values(t.Events))this.#i.on(s,a);this.#i.on(t.Events.ERROR,this.#a.bind(this));for(const s of this.#n)s(this.#i);this.#e.player.dispatch("hls-instance",{detail:this.#i}),this.#i.attachMedia(this.#t),this.#i.on(t.Events.AUDIO_TRACK_SWITCHED,this.#o.bind(this)),this.#i.on(t.Events.LEVEL_SWITCHED,this.#c.bind(this)),this.#i.on(t.Events.LEVEL_LOADED,this.#h.bind(this)),this.#i.on(t.Events.LEVEL_UPDATED,this.#l.bind(this)),this.#i.on(t.Events.NON_NATIVE_TEXT_TRACKS_FOUND,this.#d.bind(this)),this.#i.on(t.Events.CUES_PARSED,this.#u.bind(this)),this.#e.qualities[n.kW.enableAuto]=this.#v.bind(this),(0,s.listenEvent)(this.#e.qualities,"change",this.#p.bind(this)),(0,s.listenEvent)(this.#e.audioTracks,"change",this.#b.bind(this)),this.#s=(0,s.effect)(this.#y.bind(this))}#E(t,e){return new s.DOMEvent((t=>(0,s.camelToKebabCase)(t))(t),{detail:e})}#y(){if(!this.#e.$state.live())return;const t=new n.e8(this.#g.bind(this));return t.start(),t.stop.bind(t)}#g(){this.#e.$state.liveSyncPosition.set(this.#i?.liveSyncPosition??1/0)}#r(t,e){this.#e.player?.dispatch(this.#E(t,e))}#d(t,e){const i=this.#E(t,e);let s=-1;for(let r=0;r<e.tracks.length;r++){const t=e.tracks[r],a=t.subtitleTrack??t.closedCaptions,o=new n.to({id:`hls-${t.kind}-${r}`,src:a?.url,label:t.label,language:a?.lang,kind:t.kind,default:t.default});o[n.Hp.readyState]=2,o[n.Hp.onModeChange]=()=>{"showing"===o.mode?(this.#i.subtitleTrack=r,s=r):s===r&&(this.#i.subtitleTrack=-1,s=-1)},this.#e.textTracks.add(o,i)}}#u(t,e){const i=this.#i?.subtitleTrack,s=this.#e.textTracks.getById(`hls-${e.type}-${i}`);if(!s)return;const n=this.#E(t,e);for(const r of e.cues)r.positionAlign="auto",s.addCue(r,n)}#o(t,e){const i=this.#e.audioTracks[e.id];if(i){const s=this.#E(t,e);this.#e.audioTracks[n.jH.select](i,!0,s)}}#c(t,e){const i=this.#e.qualities[e.level];if(i){const s=this.#E(t,e);this.#e.qualities[n.jH.select](i,!0,s)}}#l(t,e){e.details.totalduration>0&&this.#e.$state.inferredLiveDVRWindow.set(e.details.totalduration)}#h(t,e){if(this.#e.$state.canPlay())return;const{type:i,live:r,totalduration:a,targetduration:o}=e.details,c=this.#E(t,e);this.#e.notify("stream-type-change",r?"EVENT"===i&&Number.isFinite(a)&&o>=10?"live:dvr":"live":"on-demand",c),this.#e.notify("duration-change",a,c);const h=this.#i.media;-1===this.#i.currentLevel&&this.#e.qualities[n.kW.setAuto](!0,c);for(const s of this.#i.audioTracks){const t={id:s.id.toString(),label:s.name,language:s.lang||"",kind:"main"};this.#e.audioTracks[n.jH.add](t,c)}for(const s of this.#i.levels){const t={id:s.id?.toString()??s.height+"p",width:s.width,height:s.height,codec:s.codecSet,bitrate:s.bitrate};this.#e.qualities[n.jH.add](t,c)}h.dispatchEvent(new s.DOMEvent("canplay",{trigger:c}))}#a(t,e){if(e.fatal)if("mediaError"===e.type)this.#i?.recoverMediaError();else this.#f(e.error)}#f(t){this.#e.notify("error",{message:t.message,code:1,error:t})}#v(){this.#i&&(this.#i.currentLevel=-1)}#p(){const{qualities:t}=this.#e;this.#i&&!t.auto&&(this.#i[t.switch+"Level"]=t.selectedIndex,n.G_&&(this.#t.currentTime=this.#t.currentTime))}#b(){const{audioTracks:t}=this.#e;this.#i&&this.#i.audioTrack!==t.selectedIndex&&(this.#i.audioTrack=t.selectedIndex)}onInstance(t){return this.#n.add(t),()=>this.#n.delete(t)}loadSource(t){(0,s.isString)(t.src)&&this.#i?.loadSource(t.src)}destroy(){this.#i?.destroy(),this.#i=null,this.#s?.(),this.#s=null}}class a{#S;#e;#L;constructor(t,e,i){this.#S=t,this.#e=e,this.#L=i,this.#x()}async#x(){const t={onLoadStart:this.#k.bind(this),onLoaded:this.#w.bind(this),onLoadError:this.#T.bind(this)};let e=await async function(t,e={}){if(!(0,s.isString)(t))return;e.onLoadStart?.();try{if(await(0,n.k0)(t),!(0,s.isFunction)(window.Hls))throw Error("");const i=window.Hls;return e.onLoaded?.(i),i}catch(i){e.onLoadError?.(i)}return}(this.#S,t);if((0,s.isUndefined)(e)&&!(0,s.isString)(this.#S)&&(e=await async function(t,e={}){if((0,s.isUndefined)(t))return;if(e.onLoadStart?.(),t.prototype&&t.prototype!==Function)return e.onLoaded?.(t),t;try{const i=(await t())?.default;if(!i||!i.isSupported)throw Error("");return e.onLoaded?.(i),i}catch(i){e.onLoadError?.(i)}return}(this.#S,t)),!e)return null;if(!e.isSupported()){const t="[vidstack] `hls.js` is not supported in this environment";return this.#e.player.dispatch(new s.DOMEvent("hls-unsupported")),this.#e.notify("error",{message:t,code:4}),null}return e}#k(){this.#e.player.dispatch(new s.DOMEvent("hls-lib-load-start"))}#w(t){this.#e.player.dispatch(new s.DOMEvent("hls-lib-loaded",{detail:t})),this.#L(t)}#T(t){const e=(0,n.rv)(t);this.#e.player.dispatch(new s.DOMEvent("hls-lib-load-error",{detail:e})),this.#e.notify("error",{message:e.message,code:4,error:e})}}class o extends n.Nw{$$PROVIDER_TYPE="HLS";#m=null;#D=new r(this.video,this.ctx);get ctor(){return this.#m}get instance(){return this.#D.instance}static supported=(0,n.m0)();get type(){return"hls"}get canLiveSync(){return!0}#O="https://cdn.jsdelivr.net/npm/hls.js@^1.5.0/dist/hls.min.js";get config(){return this.#D.config}set config(t){this.#D.config=t}get library(){return this.#O}set library(t){this.#O=t}preconnect(){(0,s.isString)(this.#O)&&(0,n.TB)(this.#O)}setup(){super.setup(),new a(this.#O,this.ctx,(t=>{this.#m=t,this.#D.setup(t),this.ctx.notify("provider-setup",this);const e=(0,s.peek)(this.ctx.$state.source);e&&this.loadSource(e)}))}async loadSource(t,e){(0,s.isString)(t.src)?(this.media.preload=e||"",this.appendSource(t,"application/x-mpegurl"),this.#D.loadSource(t),this.currentSrc=t):this.removeSource()}onInstance(t){const e=this.#D.instance;return e&&t(e),this.#D.onInstance(t)}destroy(){this.#D.destroy()}}}}]);