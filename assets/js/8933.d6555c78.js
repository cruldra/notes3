"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[8933],{1190:(e,t,r)=>{r.d(t,{Q:()=>te});var n,a=r(6070),o=r(758),i=r(3526);function u(e,t){var r={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&t.indexOf(n)<0&&(r[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols){var a=0;for(n=Object.getOwnPropertySymbols(e);a<n.length;a++)t.indexOf(n[a])<0&&Object.prototype.propertyIsEnumerable.call(e,n[a])&&(r[n[a]]=e[n[a]])}return r}function l(){}function s(e){return!!(e||"").match(/\d/)}function c(e){return null==e}function d(e){return c(e)||function(e){return"number"==typeof e&&isNaN(e)}(e)||"number"==typeof e&&!isFinite(e)}function f(e){return e.replace(/[-[\]/{}()*+?.\\^$|]/g,"\\$&")}function v(e,t){void 0===t&&(t=!0);var r="-"===e[0],n=r&&t,a=(e=e.replace("-","")).split(".");return{beforeDecimal:a[0],afterDecimal:a[1]||"",hasNegation:r,addNegation:n}}function p(e,t,r){for(var n="",a=r?"0":"",o=0;o<=t-1;o++)n+=e[o]||a;return n}function m(e,t){return Array(t+1).join(e)}function g(e){var t=e+"",r="-"===t[0]?"-":"";r&&(t=t.substring(1));var n=t.split(/[eE]/g),a=n[0],o=n[1];if(!(o=Number(o)))return r+a;var i=1+o,u=(a=a.replace(".","")).length;return i<0?a="0."+m("0",Math.abs(i))+a:i>=u?a+=m("0",i-u):a=(a.substring(0,i)||"0")+"."+a.substring(i),r+a}function h(e,t,r){if(-1!==["","-"].indexOf(e))return e;var n=(-1!==e.indexOf(".")||r)&&t,a=v(e),o=a.beforeDecimal,i=a.afterDecimal,u=a.hasNegation,l=parseFloat("0."+(i||"0")),s=(i.length<=t?"0."+i:l.toFixed(t)).split("."),c=o;return o&&Number(s[0])&&(c=o.split("").reverse().reduce((function(e,t,r){return e.length>r?(Number(e[0])+Number(t)).toString()+e.substring(1,e.length):t+e}),s[0])),""+(u?"-":"")+c+(n?".":"")+p(s[1]||"",t,r)}function b(e,t){if(e.value=e.value,null!==e){if(e.createTextRange){var r=e.createTextRange();return r.move("character",t),r.select(),!0}return e.selectionStart||0===e.selectionStart?(e.focus(),e.setSelectionRange(t,t),!0):(e.focus(),!1)}}!function(e){e.event="event",e.props="prop"}(n||(n={}));var y,S,w,x=(y=function(e,t){for(var r=0,n=0,a=e.length,o=t.length;e[r]===t[r]&&r<a;)r++;for(;e[a-1-n]===t[o-1-n]&&o-n>r&&a-n>r;)n++;return{from:{start:r,end:a-n},to:{start:r,end:o-n}}},w=void 0,function(){for(var e=[],t=arguments.length;t--;)e[t]=arguments[t];return S&&e.length===S.length&&e.every((function(e,t){return e===S[t]}))?w:(S=e,w=y.apply(void 0,e))});function C(e){return Math.max(e.selectionStart,e.selectionEnd)}function N(e){return{from:{start:0,end:0},to:{start:0,end:e.length},lastValue:""}}function V(e){var t=e.currentValue,r=e.formattedValue,n=e.currentValueIndex,a=e.formattedValueIndex;return t[n]===r[a]}function D(e,t,r,n){var a,o,i,u=e.length;if(a=t,o=0,i=u,t=Math.min(Math.max(a,o),i),"left"===n){for(;t>=0&&!r[t];)t--;-1===t&&(t=r.indexOf(!0))}else{for(;t<=u&&!r[t];)t++;t>u&&(t=r.lastIndexOf(!0))}return-1===t&&(t=u),t}function E(e){for(var t=Array.from({length:e.length+1}).map((function(){return!0})),r=0,n=t.length;r<n;r++)t[r]=Boolean(s(e[r])||s(e[r-1]));return t}function R(e,t,r,n,a,i){void 0===i&&(i=l);var u=function(e){var t=(0,o.useRef)(e);t.current=e;var r=(0,o.useRef)((function(){for(var e=[],r=arguments.length;r--;)e[r]=arguments[r];return t.current.apply(t,e)}));return r.current}((function(e,t){var r,o;return d(e)?(o="",r=""):"number"==typeof e||t?(o="number"==typeof e?g(e):e,r=n(o)):(o=a(e,void 0),r=n(o)),{formattedValue:r,numAsString:o}})),s=(0,o.useState)((function(){return u(c(e)?t:e,r)})),f=s[0],v=s[1],p=e,m=r;c(e)&&(p=f.numAsString,m=!0);var h=u(p,m);return(0,o.useMemo)((function(){v(h)}),[h.formattedValue]),[f,function(e,t){e.formattedValue!==f.formattedValue&&v({formattedValue:e.formattedValue,numAsString:e.value}),i(e,t)}]}function T(e){return e.replace(/[^0-9]/g,"")}function I(e){return e}function M(e){var t=e.type;void 0===t&&(t="text");var r=e.displayType;void 0===r&&(r="input");var a=e.customInput,i=e.renderText,c=e.getInputRef,d=e.format;void 0===d&&(d=I);var f=e.removeFormatting;void 0===f&&(f=T);var v=e.defaultValue,p=e.valueIsNumericString,m=e.onValueChange,g=e.isAllowed,h=e.onChange;void 0===h&&(h=l);var y=e.onKeyDown;void 0===y&&(y=l);var S=e.onMouseUp;void 0===S&&(S=l);var w=e.onFocus;void 0===w&&(w=l);var N=e.onBlur;void 0===N&&(N=l);var M=e.value,O=e.getCaretBoundary;void 0===O&&(O=E);var P=e.isValidInputCharacter;void 0===P&&(P=s);var A=e.isCharacterSame,j=u(e,["type","displayType","customInput","renderText","getInputRef","format","removeFormatting","defaultValue","valueIsNumericString","onValueChange","isAllowed","onChange","onKeyDown","onMouseUp","onFocus","onBlur","value","getCaretBoundary","isValidInputCharacter","isCharacterSame"]),z=R(M,v,Boolean(p),d,f,m),B=z[0],F=B.formattedValue,L=B.numAsString,_=z[1],k=(0,o.useRef)(),W=(0,o.useRef)({formattedValue:F,numAsString:L}),Z=function(e,t){W.current={formattedValue:e.formattedValue,numAsString:e.value},_(e,t)},K=(0,o.useState)(!1),H=K[0],Y=K[1],$=(0,o.useRef)(null),G=(0,o.useRef)({setCaretTimeout:null,focusTimeout:null});(0,o.useEffect)((function(){return Y(!0),function(){clearTimeout(G.current.setCaretTimeout),clearTimeout(G.current.focusTimeout)}}),[]);var U=d,q=function(e,t){var r=parseFloat(t);return{formattedValue:e,value:t,floatValue:isNaN(r)?void 0:r}},X=function(e,t,r){0===e.selectionStart&&e.selectionEnd===e.value.length||(b(e,t),G.current.setCaretTimeout=setTimeout((function(){e.value===r&&e.selectionStart!==t&&b(e,t)}),0))},Q=function(e,t,r){return D(e,t,O(e),r)},J=function(e,t,r){var n=O(t),a=function(e,t,r,n,a,o,i){void 0===i&&(i=V);var u=a.findIndex((function(e){return e})),l=e.slice(0,u);t||r.startsWith(l)||(t=l,r=l+r,n+=l.length);for(var s=r.length,c=e.length,d={},f=new Array(s),v=0;v<s;v++){f[v]=-1;for(var p=0,m=c;p<m;p++)if(i({currentValue:r,lastValue:t,formattedValue:e,currentValueIndex:v,formattedValueIndex:p})&&!0!==d[p]){f[v]=p,d[p]=!0;break}}for(var g=n;g<s&&(-1===f[g]||!o(r[g]));)g++;var h=g===s||-1===f[g]?c:f[g];for(g=n-1;g>0&&-1===f[g];)g--;var b=-1===g||-1===f[g]?0:f[g]+1;return b>h?h:n-b<h-n?b:h}(t,F,e,r,n,P,A);return a=D(t,a,n)};(0,o.useEffect)((function(){var e=W.current,t=e.formattedValue,r=e.numAsString;F===t&&L===r||Z(q(F,L),{event:void 0,source:n.props})}),[F,L]);var ee=$.current?C($.current):void 0;("undefined"!=typeof window?o.useLayoutEffect:o.useEffect)((function(){var e=$.current;if(F!==W.current.formattedValue&&e){var t=J(W.current.formattedValue,F,ee);e.value=F,X(e,t,F)}}),[F]);var te=function(e,t,r){var n=t.target,a=k.current?function(e,t){var r=Math.min(e.selectionStart,t);return{from:{start:r,end:e.selectionEnd},to:{start:r,end:t}}}(k.current,n.selectionEnd):x(F,e),o=Object.assign(Object.assign({},a),{lastValue:F}),i=f(e,o),u=U(i);if(i=f(u,void 0),g&&!g(q(u,i))){var l=t.target,s=C(l),c=J(e,F,s);return l.value=F,X(l,c,F),!1}return function(e){var t=e.formattedValue;void 0===t&&(t="");var r,n=e.input,a=e.source,o=e.event,i=e.numAsString;if(n){var u=e.inputValue||n.value,l=C(n);n.value=t,void 0!==(r=J(u,t,l))&&X(n,r,t)}t!==F&&Z(q(t,i),{event:o,source:a})}({formattedValue:u,numAsString:i,inputValue:e,event:t,source:r,input:t.target}),!0},re=function(e,t){void 0===t&&(t=0);var r=e.selectionStart,n=e.selectionEnd;k.current={selectionStart:r,selectionEnd:n+t}},ne=!H||"undefined"==typeof navigator||navigator.platform&&/iPhone|iPod/.test(navigator.platform)?void 0:"numeric",ae=Object.assign({inputMode:ne},j,{type:t,value:F,onChange:function(e){var t=e.target.value;te(t,e,n.event)&&h(e),k.current=void 0},onKeyDown:function(e){var t,r=e.target,n=e.key,a=r.selectionStart,o=r.selectionEnd,i=r.value;void 0===i&&(i=""),"ArrowLeft"===n||"Backspace"===n?t=Math.max(a-1,0):"ArrowRight"===n?t=Math.min(a+1,i.length):"Delete"===n&&(t=a);var u=0;"Delete"===n&&a===o&&(u=1);var l="ArrowLeft"===n||"ArrowRight"===n;if(void 0===t||a!==o&&!l)return y(e),void re(r,u);var s=t;l?(s=Q(i,t,"ArrowLeft"===n?"left":"right"))!==t&&e.preventDefault():"Delete"!==n||P(i[t])?"Backspace"!==n||P(i[t])||(s=Q(i,t,"left")):s=Q(i,t,"right");s!==t&&X(r,s,i),y(e),re(r,u)},onMouseUp:function(e){var t=e.target,r=function(){var e=t.selectionStart,r=t.selectionEnd,n=t.value;if(void 0===n&&(n=""),e===r){var a=Q(n,e);a!==e&&X(t,a,n)}};r(),requestAnimationFrame((function(){r()})),S(e),re(t)},onFocus:function(e){e.persist&&e.persist();var t=e.target,r=e.currentTarget;$.current=t,G.current.focusTimeout=setTimeout((function(){var n=t.selectionStart,a=t.selectionEnd,o=t.value;void 0===o&&(o="");var i=Q(o,n);i===n||0===n&&a===o.length||X(t,i,o),w(Object.assign(Object.assign({},e),{currentTarget:r}))}),0)},onBlur:function(e){$.current=null,clearTimeout(G.current.focusTimeout),clearTimeout(G.current.setCaretTimeout),N(e)}});if("text"===r)return i?o.createElement(o.Fragment,null,i(F,j)||null):o.createElement("span",Object.assign({},j,{ref:c}),F);if(a){var oe=a;return o.createElement(oe,Object.assign({},ae,{ref:c}))}return o.createElement("input",Object.assign({},ae,{ref:c}))}function O(e,t){var r=t.decimalScale,n=t.fixedDecimalScale,a=t.prefix;void 0===a&&(a="");var o=t.suffix;void 0===o&&(o="");var i=t.allowNegative,u=t.thousandsGroupStyle;if(void 0===u&&(u="thousand"),""===e||"-"===e)return e;var l=P(t),s=l.thousandSeparator,c=l.decimalSeparator,d=0!==r&&-1!==e.indexOf(".")||r&&n,f=v(e,i),m=f.beforeDecimal,g=f.afterDecimal,h=f.addNegation;return void 0!==r&&(g=p(g,r,!!n)),s&&(m=function(e,t,r){var n=function(e){switch(e){case"lakh":return/(\d+?)(?=(\d\d)+(\d)(?!\d))(\.\d+)?/g;case"wan":return/(\d)(?=(\d{4})+(?!\d))/g;default:return/(\d)(?=(\d{3})+(?!\d))/g}}(r),a=e.search(/[1-9]/);return a=-1===a?e.length:a,e.substring(0,a)+e.substring(a,e.length).replace(n,"$1"+t)}(m,s,u)),a&&(m=a+m),o&&(g+=o),h&&(m="-"+m),e=m+(d&&c||"")+g}function P(e){var t=e.decimalSeparator;void 0===t&&(t=".");var r=e.thousandSeparator,n=e.allowedDecimalSeparators;return!0===r&&(r=","),n||(n=[t,"."]),{decimalSeparator:t,thousandSeparator:r,allowedDecimalSeparators:n}}function A(e,t,r){var n;void 0===t&&(t=N(e));var a=r.allowNegative,o=r.prefix;void 0===o&&(o="");var i=r.suffix;void 0===i&&(i="");var u=r.decimalScale,l=t.from,c=t.to,d=c.start,p=c.end,m=P(r),g=m.allowedDecimalSeparators,h=m.decimalSeparator,b=e[p]===h;if(s(e)&&(e===o||e===i)&&""===t.lastValue)return e;if(p-d==1&&-1!==g.indexOf(e[d])){var y=0===u?"":h;e=e.substring(0,d)+y+e.substring(d+1,e.length)}var S=function(e,t,r){var n=!1,a=!1;o.startsWith("-")?n=!1:e.startsWith("--")?(n=!1,a=!0):i.startsWith("-")&&e.length===i.length?n=!1:"-"===e[0]&&(n=!0);var u=n?1:0;return a&&(u=2),u&&(e=e.substring(u),t-=u,r-=u),{value:e,start:t,end:r,hasNegation:n}},w=S(e,d,p),x=w.hasNegation;e=(n=w).value,d=n.start,p=n.end;var C=S(t.lastValue,l.start,l.end),V=C.start,D=C.end,E=C.value,R=e.substring(d,p);!(e.length&&E.length&&(V>E.length-i.length||D<o.length))||R&&i.startsWith(R)||(e=E);var T=0;e.startsWith(o)?T+=o.length:d<o.length&&(T=d),p-=T;var I=(e=e.substring(T)).length,M=e.length-i.length;e.endsWith(i)?I=M:(p>M||p>e.length-i.length)&&(I=p),e=e.substring(0,I),e=function(e,t){void 0===e&&(e="");var r=new RegExp("(-)"),n=new RegExp("(-)(.)*(-)"),a=r.test(e),o=n.test(e);return e=e.replace(/-/g,""),a&&!o&&t&&(e="-"+e),e}(x?"-"+e:e,a),e=(e.match(function(e,t){return new RegExp("(^-)|[0-9]|"+f(e),t?"g":void 0)}(h,!0))||[]).join("");var O=e.indexOf(h),A=v(e=e.replace(new RegExp(f(h),"g"),(function(e,t){return t===O?".":""})),a),j=A.beforeDecimal,z=A.afterDecimal,B=A.addNegation;return c.end-c.start<l.end-l.start&&""===j&&b&&!parseFloat(z)&&(e=B?"-":""),e}function j(e){e=function(e){var t=P(e),r=t.thousandSeparator,n=t.decimalSeparator,a=e.prefix;void 0===a&&(a="");var o=e.allowNegative;if(void 0===o&&(o=!0),r===n)throw new Error("\n        Decimal separator can't be same as thousand separator.\n        thousandSeparator: "+r+' (thousandSeparator = {true} is same as thousandSeparator = ",")\n        decimalSeparator: '+n+" (default value for decimalSeparator is .)\n     ");return a.startsWith("-")&&o&&(console.error("\n      Prefix can't start with '-' when allowNegative is true.\n      prefix: "+a+"\n      allowNegative: "+o+"\n    "),o=!1),Object.assign(Object.assign({},e),{allowNegative:o})}(e);e.decimalSeparator,e.allowedDecimalSeparators,e.thousandsGroupStyle;var t=e.suffix,r=e.allowNegative,a=e.allowLeadingZeros,o=e.onKeyDown;void 0===o&&(o=l);var i=e.onBlur;void 0===i&&(i=l);var f=e.thousandSeparator,v=e.decimalScale,p=e.fixedDecimalScale,m=e.prefix;void 0===m&&(m="");var y=e.defaultValue,S=e.value,w=e.valueIsNumericString,C=e.onValueChange,N=u(e,["decimalSeparator","allowedDecimalSeparators","thousandsGroupStyle","suffix","allowNegative","allowLeadingZeros","onKeyDown","onBlur","thousandSeparator","decimalScale","fixedDecimalScale","prefix","defaultValue","value","valueIsNumericString","onValueChange"]),V=P(e),D=V.decimalSeparator,E=V.allowedDecimalSeparators,T=function(t){return O(t,e)},I=function(t,r){return A(t,r,e)},M=c(S)?y:S,j=null!=w?w:function(e,t,r){return""===e||!(null==t?void 0:t.match(/\d/))&&!(null==r?void 0:r.match(/\d/))&&"string"==typeof e&&!isNaN(Number(e))}(M,m,t);c(S)?c(y)||(j=j||"number"==typeof y):j=j||"number"==typeof S;var z=function(e){return d(e)?e:("number"==typeof e&&(e=g(e)),j&&"number"==typeof v?h(e,v,Boolean(p)):e)},B=R(z(S),z(y),Boolean(j),T,I,C),F=B[0],L=F.numAsString,_=F.formattedValue,k=B[1];return Object.assign(Object.assign({},N),{value:_,valueIsNumericString:!1,isValidInputCharacter:function(e){return e===D||s(e)},isCharacterSame:function(e){var t=e.currentValue,r=e.lastValue,n=e.formattedValue,a=e.currentValueIndex,o=e.formattedValueIndex,i=t[a],u=n[o],l=x(r,t).to,s=function(e){return I(e).indexOf(".")+m.length};return!(0===S&&p&&v&&t[l.start]===D&&s(t)<a&&s(n)>o)&&(!!(a>=l.start&&a<l.end&&E&&E.includes(i)&&u===D)||i===u)},onValueChange:k,format:T,removeFormatting:I,getCaretBoundary:function(t){return function(e,t){var r=t.prefix;void 0===r&&(r="");var n=t.suffix;void 0===n&&(n="");var a=Array.from({length:e.length+1}).map((function(){return!0})),o="-"===e[0];a.fill(!1,0,r.length+(o?1:0));var i=e.length;return a.fill(!1,i-n.length+1,i+1),a}(t,e)},onKeyDown:function(e){var t=e.target,n=e.key,a=t.selectionStart,i=t.selectionEnd,u=t.value;if(void 0===u&&(u=""),("Backspace"===n||"Delete"===n)&&i<m.length)e.preventDefault();else if(a===i){"Backspace"===n&&"-"===u[0]&&a===m.length+1&&r&&b(t,1),v&&p&&("Backspace"===n&&u[a-1]===D?(b(t,a-1),e.preventDefault()):"Delete"===n&&u[a]===D&&e.preventDefault()),(null==E?void 0:E.includes(n))&&u[a]===D&&b(t,a+1);var l=!0===f?",":f;"Backspace"===n&&u[a-1]===l&&b(t,a-1),"Delete"===n&&u[a]===l&&b(t,a+1),o(e)}else o(e)},onBlur:function(t){var r=L;if(r.match(/\d/g)||(r=""),a||(r=function(e){if(!e)return e;var t="-"===e[0];t&&(e=e.substring(1,e.length));var r=e.split("."),n=r[0].replace(/^0+/,"")||"0",a=r[1]||"";return(t?"-":"")+n+(a?"."+a:"")}(r)),p&&v&&(r=h(r,v,p)),r!==L){var o=O(r,e);k({formattedValue:o,value:r,floatValue:parseFloat(r)},{event:t,source:n.event})}i(t)}})}function z(e){var t=j(e);return o.createElement(M,Object.assign({},t))}var B=r(327);function F(e,t,r){return void 0===t&&void 0===r?e:void 0!==t&&void 0===r?Math.max(e,t):void 0===t&&void 0!==r?Math.min(e,r):Math.min(Math.max(e,t),r)}var L=r(8645),_=r(3885),k=r(9671),W=r(3325),Z=r(1569),K=r(705),H=r(9937),Y=r(7036),$=r(247);function G({direction:e,style:t,...r}){return(0,a.jsx)("svg",{style:{width:"var(--ni-chevron-size)",height:"var(--ni-chevron-size)",transform:"up"===e?"rotate(180deg)":void 0,...t},viewBox:"0 0 15 15",fill:"none",xmlns:"http://www.w3.org/2000/svg",...r,children:(0,a.jsx)("path",{d:"M3.13523 6.15803C3.3241 5.95657 3.64052 5.94637 3.84197 6.13523L7.5 9.56464L11.158 6.13523C11.3595 5.94637 11.6759 5.95657 11.8648 6.15803C12.0536 6.35949 12.0434 6.67591 11.842 6.86477L7.84197 10.6148C7.64964 10.7951 7.35036 10.7951 7.15803 10.6148L3.15803 6.86477C2.95657 6.67591 2.94637 6.35949 3.13523 6.15803Z",fill:"currentColor",fillRule:"evenodd",clipRule:"evenodd"})})}var U={root:"m_e2f5cd4e",controls:"m_95e17d22",control:"m_80b4b171"};const q=/^(0\.0*|-0(\.0*)?)$/,X=/^-?0\d+(\.\d+)?\.?$/;function Q(e,t,r){if(void 0===e)return!0;return(void 0===t||e>=t)&&(void 0===r||e<=r)}const J={step:1,clampBehavior:"blur",allowDecimal:!0,allowNegative:!0,withKeyboardEvents:!0,allowLeadingZeros:!0,trimLeadingZeroesOnBlur:!0,startValue:0},ee=(0,k.V)(((e,{size:t})=>({controls:{"--ni-chevron-size":(0,_.YC)(t,"ni-chevron-size")}}))),te=(0,H.P9)(((e,t)=>{const r=(0,K.Y)("NumberInput",J,e),{className:n,classNames:u,styles:l,unstyled:s,vars:c,onChange:d,onValueChange:f,value:v,defaultValue:p,max:m,min:g,step:h,hideControls:b,rightSection:y,isAllowed:S,clampBehavior:w,onBlur:x,allowDecimal:C,decimalScale:N,onKeyDown:V,onKeyDownCapture:D,handlersRef:E,startValue:R,disabled:T,rightSectionPointerEvents:I,allowNegative:M,readOnly:O,size:P,rightSectionWidth:A,stepHoldInterval:j,stepHoldDelay:_,allowLeadingZeros:k,withKeyboardEvents:H,trimLeadingZeroesOnBlur:te,...re}=r,ne=(0,Z.I)({name:"NumberInput",classes:U,props:r,classNames:u,styles:l,unstyled:s,vars:c,varsResolver:ee}),{resolvedClassNames:ae,resolvedStyles:oe}=(0,W.Y)({classNames:u,styles:l,props:r}),[ie,ue]=(0,B.Z)({value:v,defaultValue:p,onChange:d}),le=void 0!==_&&void 0!==j,se=(0,o.useRef)(null),ce=(0,o.useRef)(null),de=(0,o.useRef)(0),fe=e=>{const t=String(e).match(/(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/);return t?Math.max(0,(t[1]?t[1].length:0)-(t[2]?+t[2]:0)):0},ve=e=>{se.current&&void 0!==e&&se.current.setSelectionRange(e,e)},pe=(0,o.useRef)();pe.current=()=>{let e;const t=fe(ie),r=fe(h),n=Math.max(t,r),a=10**n;if("number"!=typeof ie||Number.isNaN(ie))e=F(R,g,m);else if(void 0!==m){const t=(Math.round(ie*a)+Math.round(h*a))/a;e=t<=m?t:m}else e=(Math.round(ie*a)+Math.round(h*a))/a;const o=e.toFixed(n);ue(parseFloat(o)),f?.({floatValue:parseFloat(o),formattedValue:o,value:o},{source:"increment"}),setTimeout((()=>ve(se.current?.value.length)),0)};const me=(0,o.useRef)();me.current=()=>{let e;const t=void 0!==g?g:M?Number.MIN_SAFE_INTEGER:0,r=fe(ie),n=fe(h),a=Math.max(r,n),o=10**a;if("number"!=typeof ie||Number.isNaN(ie))e=F(R,t,m);else{const r=(Math.round(ie*o)-Math.round(h*o))/o;e=void 0!==t&&r<t?t:r}const i=e.toFixed(a);ue(parseFloat(i)),f?.({floatValue:parseFloat(i),formattedValue:i,value:i},{source:"decrement"}),setTimeout((()=>ve(se.current?.value.length)),0)};(0,L.bl)(E,{increment:pe.current,decrement:me.current});const ge=e=>{e?pe.current():me.current(),de.current+=1},he=e=>{if(ge(e),le){const t="number"==typeof j?j:j(de.current);ce.current=window.setTimeout((()=>he(e)),t)}},be=(e,t)=>{e.preventDefault(),se.current?.focus(),ge(t),le&&(ce.current=window.setTimeout((()=>he(t)),_))},ye=()=>{ce.current&&window.clearTimeout(ce.current),ce.current=null,de.current=0},Se=(0,a.jsxs)("div",{...ne("controls"),children:[(0,a.jsx)($.N,{...ne("control"),tabIndex:-1,"aria-hidden":!0,disabled:T||"number"==typeof ie&&void 0!==m&&ie>=m,mod:{direction:"up"},onMouseDown:e=>e.preventDefault(),onPointerDown:e=>{be(e,!0)},onPointerUp:ye,onPointerLeave:ye,children:(0,a.jsx)(G,{direction:"up"})}),(0,a.jsx)($.N,{...ne("control"),tabIndex:-1,"aria-hidden":!0,disabled:T||"number"==typeof ie&&void 0!==g&&ie<=g,mod:{direction:"down"},onMouseDown:e=>e.preventDefault(),onPointerDown:e=>{be(e,!1)},onPointerUp:ye,onPointerLeave:ye,children:(0,a.jsx)(G,{direction:"down"})})]});return(0,a.jsx)(Y.O,{component:z,allowNegative:M,className:(0,i.A)(U.root,n),size:P,...re,readOnly:O,disabled:T,value:ie,getInputRef:(0,L.pc)(t,se),onValueChange:(e,t)=>{"event"===t.source&&ue(!function(e,t){return("number"==typeof e?e<Number.MAX_SAFE_INTEGER:!Number.isNaN(Number(e)))&&!Number.isNaN(e)&&t.toString().replace(".","").length<14&&""!==t}(e.floatValue,e.value)||q.test(e.value)||k&&X.test(e.value)?e.value:e.floatValue),f?.(e,t)},rightSection:b||O?y:y||Se,classNames:ae,styles:oe,unstyled:s,__staticSelector:"NumberInput",decimalScale:C?N:0,onKeyDown:e=>{V?.(e),!O&&H&&("ArrowUp"===e.key&&(e.preventDefault(),pe.current()),"ArrowDown"===e.key&&(e.preventDefault(),me.current()))},onKeyDownCapture:e=>{if(D?.(e),"Backspace"===e.key){const t=se.current;0===t.selectionStart&&t.selectionStart===t.selectionEnd&&(e.preventDefault(),window.setTimeout((()=>ve(0)),0))}},rightSectionPointerEvents:I??(T?"none":void 0),rightSectionWidth:A??`var(--ni-right-section-width-${P||"sm"})`,allowLeadingZeros:k,onBlur:e=>{if(x?.(e),"blur"===w&&"number"==typeof ie){F(ie,g,m)!==ie&&ue(F(ie,g,m))}if(te&&"string"==typeof ie&&fe(ie)<15){const e=ie.replace(/^0+/,""),t=parseFloat(e);ue(Number.isNaN(t)||t>Number.MAX_SAFE_INTEGER?e:F(t,g,m))}},isAllowed:e=>"strict"===w?S?S(e)&&Q(e.floatValue,g,m):Q(e.floatValue,g,m):!S||S(e)})}));te.classes={...Y.O.classes,...U},te.displayName="@mantine/core/NumberInput"},3121:(e,t,r)=>{r.d(t,{y:()=>S});var n=r(6070),a=r(3526),o=r(2147),i=r(327),u=(r(758),r(3885)),l=r(9671),s=r(3325),c=r(1569),d=r(705),f=r(9744),v=r(9937),p=r(1546),m=r(2520),g=r(7036);var h={root:"m_f61ca620",input:"m_ccf8da4c",innerInput:"m_f2d85dd2",visibilityToggle:"m_b1072d44"};const b={visibilityToggleIcon:({reveal:e})=>(0,n.jsx)("svg",{viewBox:"0 0 15 15",fill:"none",xmlns:"http://www.w3.org/2000/svg",style:{width:"var(--psi-icon-size)",height:"var(--psi-icon-size)"},children:(0,n.jsx)("path",{d:e?"M13.3536 2.35355C13.5488 2.15829 13.5488 1.84171 13.3536 1.64645C13.1583 1.45118 12.8417 1.45118 12.6464 1.64645L10.6828 3.61012C9.70652 3.21671 8.63759 3 7.5 3C4.30786 3 1.65639 4.70638 0.0760002 7.23501C-0.0253338 7.39715 -0.0253334 7.60288 0.0760014 7.76501C0.902945 9.08812 2.02314 10.1861 3.36061 10.9323L1.64645 12.6464C1.45118 12.8417 1.45118 13.1583 1.64645 13.3536C1.84171 13.5488 2.15829 13.5488 2.35355 13.3536L4.31723 11.3899C5.29348 11.7833 6.36241 12 7.5 12C10.6921 12 13.3436 10.2936 14.924 7.76501C15.0253 7.60288 15.0253 7.39715 14.924 7.23501C14.0971 5.9119 12.9769 4.81391 11.6394 4.06771L13.3536 2.35355ZM9.90428 4.38861C9.15332 4.1361 8.34759 4 7.5 4C4.80285 4 2.52952 5.37816 1.09622 7.50001C1.87284 8.6497 2.89609 9.58106 4.09974 10.1931L9.90428 4.38861ZM5.09572 10.6114L10.9003 4.80685C12.1039 5.41894 13.1272 6.35031 13.9038 7.50001C12.4705 9.62183 10.1971 11 7.5 11C6.65241 11 5.84668 10.8639 5.09572 10.6114Z":"M7.5 11C4.80285 11 2.52952 9.62184 1.09622 7.50001C2.52952 5.37816 4.80285 4 7.5 4C10.1971 4 12.4705 5.37816 13.9038 7.50001C12.4705 9.62183 10.1971 11 7.5 11ZM7.5 3C4.30786 3 1.65639 4.70638 0.0760002 7.23501C-0.0253338 7.39715 -0.0253334 7.60288 0.0760014 7.76501C1.65639 10.2936 4.30786 12 7.5 12C10.6921 12 13.3436 10.2936 14.924 7.76501C15.0253 7.60288 15.0253 7.39715 14.924 7.23501C13.3436 4.70638 10.6921 3 7.5 3ZM7.5 9.5C8.60457 9.5 9.5 8.60457 9.5 7.5C9.5 6.39543 8.60457 5.5 7.5 5.5C6.39543 5.5 5.5 6.39543 5.5 7.5C5.5 8.60457 6.39543 9.5 7.5 9.5Z",fill:"currentColor",fillRule:"evenodd",clipRule:"evenodd"})})},y=(0,l.V)(((e,{size:t})=>({root:{"--psi-icon-size":(0,u.YC)(t,"psi-icon-size"),"--psi-button-size":(0,u.YC)(t,"psi-button-size")}}))),S=(0,v.P9)(((e,t)=>{const r=(0,d.Y)("PasswordInput",b,e),{classNames:u,className:l,style:v,styles:g,unstyled:S,vars:w,required:x,error:C,leftSection:N,disabled:V,id:D,variant:E,inputContainer:R,description:T,label:I,size:M,errorProps:O,descriptionProps:P,labelProps:A,withAsterisk:j,inputWrapperOrder:z,wrapperProps:B,radius:F,rightSection:L,rightSectionWidth:_,rightSectionPointerEvents:k,leftSectionWidth:W,visible:Z,defaultVisible:K,onVisibilityChange:H,visibilityToggleIcon:Y,visibilityToggleButtonProps:$,rightSectionProps:G,leftSectionProps:U,leftSectionPointerEvents:q,withErrorStyles:X,mod:Q,...J}=r,ee=(0,o.B)(D),[te,re]=(0,i.Z)({value:Z,defaultValue:K,finalValue:!1,onChange:H}),ne=()=>re(!te),ae=(0,c.I)({name:"PasswordInput",classes:h,props:r,className:l,style:v,classNames:u,styles:g,unstyled:S,vars:w,varsResolver:y}),{resolvedClassNames:oe,resolvedStyles:ie}=(0,s.Y)({classNames:u,styles:g,props:r}),{styleProps:ue,rest:le}=(0,f.j)(J),se=Y,ce=(0,n.jsx)(p.M,{...ae("visibilityToggle"),disabled:V,radius:F,"aria-hidden":!$,tabIndex:-1,...$,variant:"subtle",color:"gray",unstyled:S,onTouchEnd:e=>{e.preventDefault(),$?.onTouchEnd?.(e),ne()},onMouseDown:e=>{e.preventDefault(),$?.onMouseDown?.(e),ne()},onKeyDown:e=>{$?.onKeyDown?.(e)," "===e.key&&(e.preventDefault(),ne())},children:(0,n.jsx)(se,{reveal:te})});return(0,n.jsx)(m.p.Wrapper,{required:x,id:ee,label:I,error:C,description:T,size:M,classNames:oe,styles:ie,__staticSelector:"PasswordInput",errorProps:O,descriptionProps:P,unstyled:S,withAsterisk:j,inputWrapperOrder:z,inputContainer:R,variant:E,labelProps:{...A,htmlFor:ee},mod:Q,...ae("root"),...ue,...B,children:(0,n.jsx)(m.p,{component:"div",error:C,leftSection:N,size:M,classNames:{...oe,input:(0,a.A)(h.input,oe.input)},styles:ie,radius:F,disabled:V,__staticSelector:"PasswordInput",rightSectionWidth:_,rightSection:L??ce,variant:E,unstyled:S,leftSectionWidth:W,rightSectionPointerEvents:k||"all",rightSectionProps:G,leftSectionProps:U,leftSectionPointerEvents:q,withAria:!1,withErrorStyles:X,children:(0,n.jsx)("input",{required:x,"data-invalid":!!C||void 0,"data-with-left-section":!!N||void 0,...ae("innerInput"),disabled:V,id:ee,ref:t,...le,autoComplete:le.autoComplete||"off",type:te?"text":"password"})})})}));S.classes={...g.O.classes,...h},S.displayName="@mantine/core/PasswordInput"},9178:(e,t,r)=>{r.d(t,{T:()=>D});var n=r(6070),a=r(5890),o=r(5045),i=r(758);const u=i.useLayoutEffect;var l=function(e,t){"function"!=typeof e?e.current=t:e(t)};const s=function(e,t){var r=(0,i.useRef)();return(0,i.useCallback)((function(n){e.current=n,r.current&&l(r.current,null),r.current=t,t&&l(t,n)}),[t])};var c={"min-height":"0","max-height":"none",height:"0",visibility:"hidden",overflow:"hidden",position:"absolute","z-index":"-1000",top:"0",right:"0",display:"block"},d=function(e){Object.keys(c).forEach((function(t){e.style.setProperty(t,c[t],"important")}))},f=null,v=function(e,t){var r=e.scrollHeight;return"border-box"===t.sizingStyle.boxSizing?r+t.borderSize:r-t.paddingSize};var p=function(){},m=["borderBottomWidth","borderLeftWidth","borderRightWidth","borderTopWidth","boxSizing","fontFamily","fontSize","fontStyle","fontWeight","letterSpacing","lineHeight","paddingBottom","paddingLeft","paddingRight","paddingTop","tabSize","textIndent","textRendering","textTransform","width","wordBreak"],g=!!document.documentElement.currentStyle,h=function(e){var t=window.getComputedStyle(e);if(null===t)return null;var r,n=(r=t,m.reduce((function(e,t){return e[t]=r[t],e}),{})),a=n.boxSizing;return""===a?null:(g&&"border-box"===a&&(n.width=parseFloat(n.width)+parseFloat(n.borderRightWidth)+parseFloat(n.borderLeftWidth)+parseFloat(n.paddingRight)+parseFloat(n.paddingLeft)+"px"),{sizingStyle:n,paddingSize:parseFloat(n.paddingBottom)+parseFloat(n.paddingTop),borderSize:parseFloat(n.borderBottomWidth)+parseFloat(n.borderTopWidth)})};function b(e,t,r){var n,a,o=(n=r,a=i.useRef(n),u((function(){a.current=n})),a);i.useLayoutEffect((function(){var r=function(e){return o.current(e)};if(e)return e.addEventListener(t,r),function(){return e.removeEventListener(t,r)}}),[])}var y=["cacheMeasurements","maxRows","minRows","onChange","onHeightChange"],S=function(e,t){var r=e.cacheMeasurements,n=e.maxRows,u=e.minRows,l=e.onChange,c=void 0===l?p:l,m=e.onHeightChange,g=void 0===m?p:m,S=(0,o.A)(e,y),w=void 0!==S.value,x=i.useRef(null),C=s(x,t),N=i.useRef(0),V=i.useRef(),D=function(){var e=x.current,t=r&&V.current?V.current:h(e);if(t){V.current=t;var a=function(e,t,r,n){void 0===r&&(r=1),void 0===n&&(n=1/0),f||((f=document.createElement("textarea")).setAttribute("tabindex","-1"),f.setAttribute("aria-hidden","true"),d(f)),null===f.parentNode&&document.body.appendChild(f);var a=e.paddingSize,o=e.borderSize,i=e.sizingStyle,u=i.boxSizing;Object.keys(i).forEach((function(e){var t=e;f.style[t]=i[t]})),d(f),f.value=t;var l=v(f,e);f.value=t,l=v(f,e),f.value="x";var s=f.scrollHeight-a,c=s*r;"border-box"===u&&(c=c+a+o),l=Math.max(c,l);var p=s*n;return"border-box"===u&&(p=p+a+o),[l=Math.min(p,l),s]}(t,e.value||e.placeholder||"x",u,n),o=a[0],i=a[1];N.current!==o&&(N.current=o,e.style.setProperty("height",o+"px","important"),g(o,{rowHeight:i}))}};return i.useLayoutEffect(D),b(window,"resize",D),function(e){b(document.fonts,"loadingdone",e)}(D),i.createElement("textarea",(0,a.A)({},S,{onChange:function(e){w||D(),c(e)},ref:C}))},w=i.forwardRef(S);var x=r(705),C=r(9937),N=r(7036);const V={},D=(0,C.P9)(((e,t)=>{const{autosize:r,maxRows:a,minRows:o,__staticSelector:i,resize:u,...l}=(0,x.Y)("Textarea",V,e),s=r&&"test"!=("undefined"!=typeof process&&process.env?"production":"development"),c=s?{maxRows:a,minRows:o}:{};return(0,n.jsx)(N.O,{component:s?w:"textarea",ref:t,...l,__staticSelector:i||"Textarea",multiline:!0,"data-no-overflow":r&&void 0===a||void 0,__vars:{"--input-resize":u},...c})}));D.classes=N.O.classes,D.displayName="@mantine/core/Textarea"}}]);