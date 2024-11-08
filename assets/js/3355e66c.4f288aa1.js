"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[1335],{4128:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>L,contentTitle:()=>S,default:()=>q,frontMatter:()=>A,metadata:()=>r,toc:()=>R});const r=JSON.parse('{"id":"FrontEnd/React/Libraries/Mantine","title":"Mantine","description":"Mantine\u662f\u4e00\u5957\u529f\u80fd\u9f50\u5168\u7684\u7528\u6237\u754c\u9762\u7ec4\u4ef6\u5e93,\u7528\u4e8e\u6784\u5efa\u73b0\u4ee3\u7f51\u7edc\u5e94\u7528\u7a0b\u5e8f.","source":"@site/docs/FrontEnd/React/Libraries/Mantine.mdx","sourceDirName":"FrontEnd/React/Libraries","slug":"/FrontEnd/React/Libraries/Mantine","permalink":"/notes3/docs/FrontEnd/React/Libraries/Mantine","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/FrontEnd/React/Libraries/Mantine.mdx","tags":[],"version":"current","sidebarPosition":2,"frontMatter":{"sidebar_position":2},"sidebar":"frontEnd","previous":{"title":"Vidstack","permalink":"/notes3/docs/FrontEnd/React/Libraries/Vidstack"}}');var a=n(6070),s=n(5658),l=n(758),o=n(3526),u=n(9415),i=n(5557),c=n(1289),d=n(1912),p=n(473),b=n(9997);function h(e){return l.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,l.isValidElement)(e)&&function(e){const{props:t}=e;return!!t&&"object"==typeof t&&"value"in t}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function m(e){const{values:t,children:n}=e;return(0,l.useMemo)((()=>{const e=t??function(e){return h(e).map((e=>{let{props:{value:t,label:n,attributes:r,default:a}}=e;return{value:t,label:n,attributes:r,default:a}}))}(n);return function(e){const t=(0,p.XI)(e,((e,t)=>e.value===t.value));if(t.length>0)throw new Error(`Docusaurus error: Duplicate values "${t.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[t,n])}function f(e){let{value:t,tabValues:n}=e;return n.some((e=>e.value===t))}function v(e){let{queryString:t=!1,groupId:n}=e;const r=(0,i.W6)(),a=function(e){let{queryString:t=!1,groupId:n}=e;if("string"==typeof t)return t;if(!1===t)return null;if(!0===t&&!n)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return n??null}({queryString:t,groupId:n});return[(0,d.aZ)(a),(0,l.useCallback)((e=>{if(!a)return;const t=new URLSearchParams(r.location.search);t.set(a,e),r.replace({...r.location,search:t.toString()})}),[a,r])]}function g(e){const{defaultValue:t,queryString:n=!1,groupId:r}=e,a=m(e),[s,o]=(0,l.useState)((()=>function(e){let{defaultValue:t,tabValues:n}=e;if(0===n.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(t){if(!f({value:t,tabValues:n}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${t}" but none of its children has the corresponding value. Available values are: ${n.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return t}const r=n.find((e=>e.default))??n[0];if(!r)throw new Error("Unexpected error: 0 tabValues");return r.value}({defaultValue:t,tabValues:a}))),[u,i]=v({queryString:n,groupId:r}),[d,p]=function(e){let{groupId:t}=e;const n=function(e){return e?`docusaurus.tab.${e}`:null}(t),[r,a]=(0,b.Dv)(n);return[r,(0,l.useCallback)((e=>{n&&a.set(e)}),[n,a])]}({groupId:r}),h=(()=>{const e=u??d;return f({value:e,tabValues:a})?e:null})();(0,c.A)((()=>{h&&o(h)}),[h]);return{selectedValue:s,selectValue:(0,l.useCallback)((e=>{if(!f({value:e,tabValues:a}))throw new Error(`Can't select invalid tab value=${e}`);o(e),i(e),p(e)}),[i,p,a]),tabValues:a}}var x=n(1115);const j={tabList:"tabList_EhKE",tabItem:"tabItem_OtTd"};function k(e){let{className:t,block:n,selectedValue:r,selectValue:s,tabValues:l}=e;const i=[],{blockElementScrollPositionUntilNextRender:c}=(0,u.a_)(),d=e=>{const t=e.currentTarget,n=i.indexOf(t),a=l[n].value;a!==r&&(c(t),s(a))},p=e=>{let t=null;switch(e.key){case"Enter":d(e);break;case"ArrowRight":{const n=i.indexOf(e.currentTarget)+1;t=i[n]??i[0];break}case"ArrowLeft":{const n=i.indexOf(e.currentTarget)-1;t=i[n]??i[i.length-1];break}}t?.focus()};return(0,a.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,o.A)("tabs",{"tabs--block":n},t),children:l.map((e=>{let{value:t,label:n,attributes:s}=e;return(0,a.jsx)("li",{role:"tab",tabIndex:r===t?0:-1,"aria-selected":r===t,ref:e=>i.push(e),onKeyDown:p,onClick:d,...s,className:(0,o.A)("tabs__item",j.tabItem,s?.className,{"tabs__item--active":r===t}),children:n??t},t)}))})}function y(e){let{lazy:t,children:n,selectedValue:r}=e;const s=(Array.isArray(n)?n:[n]).filter(Boolean);if(t){const e=s.find((e=>e.props.value===r));return e?(0,l.cloneElement)(e,{className:(0,o.A)("margin-top--md",e.props.className)}):null}return(0,a.jsx)("div",{className:"margin-top--md",children:s.map(((e,t)=>(0,l.cloneElement)(e,{key:t,hidden:e.props.value!==r})))})}function w(e){const t=g(e);return(0,a.jsxs)("div",{className:(0,o.A)("tabs-container",j.tabList),children:[(0,a.jsx)(k,{...t,...e}),(0,a.jsx)(y,{...t,...e})]})}function E(e){const t=(0,x.A)();return(0,a.jsx)(w,{...e,children:h(e.children)},String(t))}const V={tabItem:"tabItem_KnFb"};function I(e){let{children:t,hidden:n,className:r}=e;return(0,a.jsx)("div",{role:"tabpanel",className:(0,o.A)(V.tabItem,r),hidden:n,children:t})}var T=n(1953);const N=e=>{let{}=e;return(0,a.jsx)(T.A,{placeholder:"Basic usage"})},A={sidebar_position:2},S=void 0,L={},R=[{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2}];function _(e){const t={a:"a",code:"code",h2:"h2",p:"p",pre:"pre",...(0,s.R)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsxs)(t.p,{children:[(0,a.jsx)(t.a,{href:"https://mantine.dev/",children:"Mantine"}),"\u662f\u4e00\u5957\u529f\u80fd\u9f50\u5168\u7684\u7528\u6237\u754c\u9762\u7ec4\u4ef6\u5e93,\u7528\u4e8e\u6784\u5efa\u73b0\u4ee3\u7f51\u7edc\u5e94\u7528\u7a0b\u5e8f."]}),"\n",(0,a.jsx)(t.h2,{id:"\u5b89\u88c5",children:"\u5b89\u88c5"}),"\n","\n",(0,a.jsxs)(E,{children:[(0,a.jsx)(I,{value:"npm",label:"npm",default:!0,children:(0,a.jsx)(t.pre,{children:(0,a.jsx)(t.code,{className:"language-bash",children:"npm install @mantine/core @mantine/hooks\n"})})}),(0,a.jsx)(I,{value:"pnpm",label:"pnpm",children:(0,a.jsx)(t.pre,{children:(0,a.jsx)(t.code,{className:"language-bash",children:"pnpm i @mantine/core @mantine/hooks\n"})})})]}),"\n",(0,a.jsx)(N,{})]})}function q(e={}){const{wrapper:t}={...(0,s.R)(),...e.components};return t?(0,a.jsx)(t,{...e,children:(0,a.jsx)(_,{...e})}):_(e)}}}]);