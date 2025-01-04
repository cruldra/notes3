"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[7530],{9125:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>a,contentTitle:()=>o,default:()=>m,frontMatter:()=>c,metadata:()=>t,toc:()=>l});const t=JSON.parse('{"id":"Python/Libraries/\u6570\u636e\u5e93/Alembic","title":"Alembic","description":"Alembic\u7528\u4e8e\u5b9e\u73b0\u6570\u636e\u5e93\u7248\u672c\u63a7\u5236,\u53ef\u4ee5\u548cSQLAlchemy\u548cSQLModel\u7b49ORM\u5e93\u4e00\u8d77\u4f7f\u7528.","source":"@site/docs/Python/Libraries/\u6570\u636e\u5e93/Alembic.mdx","sourceDirName":"Python/Libraries/\u6570\u636e\u5e93","slug":"/Python/Libraries/\u6570\u636e\u5e93/Alembic","permalink":"/notes3/docs/Python/Libraries/\u6570\u636e\u5e93/Alembic","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/Python/Libraries/\u6570\u636e\u5e93/Alembic.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"python","previous":{"title":"\u6570\u636e\u5e93","permalink":"/notes3/docs/category/\u6570\u636e\u5e93"},"next":{"title":"Redis","permalink":"/notes3/docs/Python/Libraries/\u6570\u636e\u5e93/Redis"}}');var s=r(6070),i=r(5658);const c={},o=void 0,a={},l=[{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2},{value:"\u4f7f\u7528\u65b9\u6cd5",id:"\u4f7f\u7528\u65b9\u6cd5",level:2}];function d(e){const n={a:"a",code:"code",h2:"h2",p:"p",pre:"pre",...(0,i.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.a,{href:"https://alembic.sqlalchemy.org/en/latest/",children:"Alembic"}),"\u7528\u4e8e\u5b9e\u73b0\u6570\u636e\u5e93\u7248\u672c\u63a7\u5236,\u53ef\u4ee5\u548c",(0,s.jsx)(n.code,{children:"SQLAlchemy"}),"\u548c",(0,s.jsx)(n.code,{children:"SQLModel"}),"\u7b49",(0,s.jsx)(n.code,{children:"ORM"}),"\u5e93\u4e00\u8d77\u4f7f\u7528."]}),"\n",(0,s.jsx)(n.h2,{id:"\u5b89\u88c5",children:"\u5b89\u88c5"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-bash",children:"pip install alembic\n"})}),"\n",(0,s.jsx)(n.h2,{id:"\u4f7f\u7528\u65b9\u6cd5",children:"\u4f7f\u7528\u65b9\u6cd5"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'\r\n# 1. \u521d\u59cb\u5316 alembic\r\nalembic init alembic\r\n\r\n# 2. \u4fee\u6539 alembic.ini \u6587\u4ef6\u4e2d\u7684\u6570\u636e\u5e93URL\r\nsqlalchemy.url = sqlite:///courses.db\r\n\r\n# 3. \u4fee\u6539 alembic/env.py\uff0c\u5bfc\u5165\u4f60\u7684\u6a21\u578b\r\nfrom your_models import Course\r\ntarget_metadata = SQLModel.metadata\r\n\r\n# 5. \u521b\u5efa\u65b0\u7684\u8fc1\u79fb\r\nalembic revision --autogenerate -m "update course model"\r\n\r\n# 6. \u5e94\u7528\u8fc1\u79fb\r\nalembic upgrade head\n'})})]})}function m(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},5658:(e,n,r)=>{r.d(n,{R:()=>c,x:()=>o});var t=r(758);const s={},i=t.createContext(s);function c(e){const n=t.useContext(i);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:c(e.components),t.createElement(i.Provider,{value:n},e.children)}}}]);