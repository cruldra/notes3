"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[4847],{3454:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>c,contentTitle:()=>l,default:()=>h,frontMatter:()=>i,metadata:()=>t,toc:()=>d});const t=JSON.parse('{"id":"FrontEnd/Vue3/Libraries/Pinia","title":"Pinia","description":"Pinia\u662fVue\u7684\u5b98\u65b9\u72b6\u6001\u7ba1\u7406\u5e93,\u4f5c\u4e3aVuex\u7684\u7ee7\u4efb\u8005.","source":"@site/docs/FrontEnd/Vue3/Libraries/Pinia.mdx","sourceDirName":"FrontEnd/Vue3/Libraries","slug":"/FrontEnd/Vue3/Libraries/Pinia","permalink":"/notes3/docs/FrontEnd/Vue3/Libraries/Pinia","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/FrontEnd/Vue3/Libraries/Pinia.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"frontEnd","previous":{"title":"\u5e38\u7528\u5e93","permalink":"/notes3/docs/category/\u5e38\u7528\u5e93-7"}}');var a=r(6070),s=r(5658),o=r(5048),u=r(1088);const i={},l=void 0,c={},d=[{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2},{value:"\u914d\u7f6e",id:"\u914d\u7f6e",level:2},{value:"\u521b\u5efa\u72b6\u6001\u4ed3\u5e93",id:"\u521b\u5efa\u72b6\u6001\u4ed3\u5e93",level:2},{value:"\u5728\u7ec4\u4ef6\u4e2d\u4f7f\u7528",id:"\u5728\u7ec4\u4ef6\u4e2d\u4f7f\u7528",level:2}];function p(e){const n={a:"a",code:"code",h2:"h2",p:"p",pre:"pre",...(0,s.R)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsxs)(n.p,{children:[(0,a.jsx)(n.a,{href:"https://pinia.vuejs.org/zh/",children:"Pinia"}),"\u662f",(0,a.jsx)(n.code,{children:"Vue"}),"\u7684\u5b98\u65b9\u72b6\u6001\u7ba1\u7406\u5e93,\u4f5c\u4e3a",(0,a.jsx)(n.code,{children:"Vuex"}),"\u7684\u7ee7\u4efb\u8005."]}),"\n",(0,a.jsx)(n.h2,{id:"\u5b89\u88c5",children:"\u5b89\u88c5"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-bash",children:"npm install pinia\n"})}),"\n",(0,a.jsx)(n.h2,{id:"\u914d\u7f6e",children:"\u914d\u7f6e"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-ts",children:"// main.ts\r\nimport { createPinia } from 'pinia'\r\nconst pinia = createPinia()\r\napp.use(pinia)\n"})}),"\n",(0,a.jsx)(n.h2,{id:"\u521b\u5efa\u72b6\u6001\u4ed3\u5e93",children:"\u521b\u5efa\u72b6\u6001\u4ed3\u5e93"}),"\n","\n",(0,a.jsxs)(o.A,{children:[(0,a.jsx)(u.A,{value:"original",label:"\u539f\u751f",default:!0,children:(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-ts",children:"// store/counter.ts\r\nimport { defineStore } from 'pinia'\r\n\r\nexport const useCounterStore = defineStore('counter', {\r\n  // \u72b6\u6001\r\n  state: () => ({\r\n    count: 0,\r\n    name: 'Eduardo'\r\n  }),\r\n\r\n  // \u8ba1\u7b97\u5c5e\u6027\r\n  getters: {\r\n    doubleCount: (state) => state.count * 2\r\n  },\r\n\r\n  // \u65b9\u6cd5\r\n  actions: {\r\n    increment() {\r\n      this.count++\r\n    },\r\n    async fetchData() {\r\n      const data = await api.get('...')\r\n      this.someData = data\r\n    }\r\n  }\r\n})\n"})})}),(0,a.jsx)(u.A,{value:"setup",label:"Setup",children:(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-ts",children:"// store/counter.ts\r\nimport { defineStore } from 'pinia'\r\n\r\nexport const useCounterStore = defineStore('counter', () => {\r\n    const count = ref(0)\r\n    const doubleCount = computed(() => count.value * 2)\r\n\r\n    function increment() {\r\n        count.value++\r\n    }\r\n\r\n    return {count, doubleCount, increment}\r\n})\n"})})})]}),"\n",(0,a.jsx)(n.h2,{id:"\u5728\u7ec4\u4ef6\u4e2d\u4f7f\u7528",children:"\u5728\u7ec4\u4ef6\u4e2d\u4f7f\u7528"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-vue",children:"<script setup>\r\nimport { useCounterStore } from '@/stores/counter'\r\n\r\nconst store = useCounterStore()\r\n\r\n// \u8bbf\u95ee state\r\nconsole.log(store.count)\r\n\r\n// \u8c03\u7528 action\r\nstore.increment()\r\n\r\n// \u4f7f\u7528 getter\r\nconsole.log(store.doubleCount)\r\n<\/script>\n"})})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,a.jsx)(n,{...e,children:(0,a.jsx)(p,{...e})}):p(e)}},1088:(e,n,r)=>{r.d(n,{A:()=>o});r(758);var t=r(3526);const a={tabItem:"tabItem_nvWs"};var s=r(6070);function o(e){let{children:n,hidden:r,className:o}=e;return(0,s.jsx)("div",{role:"tabpanel",className:(0,t.A)(a.tabItem,o),hidden:r,children:n})}},5048:(e,n,r)=>{r.d(n,{A:()=>y});var t=r(758),a=r(3526),s=r(2973),o=r(5557),u=r(7636),i=r(2310),l=r(4919),c=r(1231);function d(e){return t.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,t.isValidElement)(e)&&function(e){const{props:n}=e;return!!n&&"object"==typeof n&&"value"in n}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function p(e){const{values:n,children:r}=e;return(0,t.useMemo)((()=>{const e=n??function(e){return d(e).map((e=>{let{props:{value:n,label:r,attributes:t,default:a}}=e;return{value:n,label:r,attributes:t,default:a}}))}(r);return function(e){const n=(0,l.XI)(e,((e,n)=>e.value===n.value));if(n.length>0)throw new Error(`Docusaurus error: Duplicate values "${n.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[n,r])}function h(e){let{value:n,tabValues:r}=e;return r.some((e=>e.value===n))}function f(e){let{queryString:n=!1,groupId:r}=e;const a=(0,o.W6)(),s=function(e){let{queryString:n=!1,groupId:r}=e;if("string"==typeof n)return n;if(!1===n)return null;if(!0===n&&!r)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return r??null}({queryString:n,groupId:r});return[(0,i.aZ)(s),(0,t.useCallback)((e=>{if(!s)return;const n=new URLSearchParams(a.location.search);n.set(s,e),a.replace({...a.location,search:n.toString()})}),[s,a])]}function m(e){const{defaultValue:n,queryString:r=!1,groupId:a}=e,s=p(e),[o,i]=(0,t.useState)((()=>function(e){let{defaultValue:n,tabValues:r}=e;if(0===r.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(n){if(!h({value:n,tabValues:r}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${n}" but none of its children has the corresponding value. Available values are: ${r.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return n}const t=r.find((e=>e.default))??r[0];if(!t)throw new Error("Unexpected error: 0 tabValues");return t.value}({defaultValue:n,tabValues:s}))),[l,d]=f({queryString:r,groupId:a}),[m,b]=function(e){let{groupId:n}=e;const r=function(e){return e?`docusaurus.tab.${e}`:null}(n),[a,s]=(0,c.Dv)(r);return[a,(0,t.useCallback)((e=>{r&&s.set(e)}),[r,s])]}({groupId:a}),v=(()=>{const e=l??m;return h({value:e,tabValues:s})?e:null})();(0,u.A)((()=>{v&&i(v)}),[v]);return{selectedValue:o,selectValue:(0,t.useCallback)((e=>{if(!h({value:e,tabValues:s}))throw new Error(`Can't select invalid tab value=${e}`);i(e),d(e),b(e)}),[d,b,s]),tabValues:s}}var b=r(1760);const v={tabList:"tabList_vBCw",tabItem:"tabItem_NxBH"};var g=r(6070);function x(e){let{className:n,block:r,selectedValue:t,selectValue:o,tabValues:u}=e;const i=[],{blockElementScrollPositionUntilNextRender:l}=(0,s.a_)(),c=e=>{const n=e.currentTarget,r=i.indexOf(n),a=u[r].value;a!==t&&(l(n),o(a))},d=e=>{let n=null;switch(e.key){case"Enter":c(e);break;case"ArrowRight":{const r=i.indexOf(e.currentTarget)+1;n=i[r]??i[0];break}case"ArrowLeft":{const r=i.indexOf(e.currentTarget)-1;n=i[r]??i[i.length-1];break}}n?.focus()};return(0,g.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,a.A)("tabs",{"tabs--block":r},n),children:u.map((e=>{let{value:n,label:r,attributes:s}=e;return(0,g.jsx)("li",{role:"tab",tabIndex:t===n?0:-1,"aria-selected":t===n,ref:e=>i.push(e),onKeyDown:d,onClick:c,...s,className:(0,a.A)("tabs__item",v.tabItem,s?.className,{"tabs__item--active":t===n}),children:r??n},n)}))})}function j(e){let{lazy:n,children:r,selectedValue:s}=e;const o=(Array.isArray(r)?r:[r]).filter(Boolean);if(n){const e=o.find((e=>e.props.value===s));return e?(0,t.cloneElement)(e,{className:(0,a.A)("margin-top--md",e.props.className)}):null}return(0,g.jsx)("div",{className:"margin-top--md",children:o.map(((e,n)=>(0,t.cloneElement)(e,{key:n,hidden:e.props.value!==s})))})}function V(e){const n=m(e);return(0,g.jsxs)("div",{className:(0,a.A)("tabs-container",v.tabList),children:[(0,g.jsx)(x,{...n,...e}),(0,g.jsx)(j,{...n,...e})]})}function y(e){const n=(0,b.A)();return(0,g.jsx)(V,{...e,children:d(e.children)},String(n))}},5658:(e,n,r)=>{r.d(n,{R:()=>o,x:()=>u});var t=r(758);const a={},s=t.createContext(a);function o(e){const n=t.useContext(s);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function u(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:o(e.components),t.createElement(s.Provider,{value:n},e.children)}}}]);