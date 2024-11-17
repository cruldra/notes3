"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[4941],{4299:(e,r,s)=>{s.r(r),s.d(r,{assets:()=>u,contentTitle:()=>p,default:()=>m,frontMatter:()=>c,metadata:()=>a,toc:()=>d});const a=JSON.parse('{"id":"FrontEnd/Nodejs/Libraries/Prisma","title":"Prisma","description":"Prisma\u662f\u4e00\u4e2a\u5f00\u6e90\u7684ORM\u6846\u67b6,\u7528\u4e8e\u4e0e\u6570\u636e\u5e93\u8fdb\u884c\u4ea4\u4e92.\u5b83\u63d0\u4f9b\u4e86\u4e00\u79cd\u7b80\u5355\u800c\u5f3a\u5927\u7684\u65b9\u5f0f\u6765\u7ba1\u7406\u6570\u636e\u5e93\u8fde\u63a5\u3001\u6267\u884c\u67e5\u8be2\u548c\u5904\u7406\u6570\u636e.","source":"@site/docs/FrontEnd/Nodejs/Libraries/Prisma.mdx","sourceDirName":"FrontEnd/Nodejs/Libraries","slug":"/FrontEnd/Nodejs/Libraries/Prisma","permalink":"/notes3/docs/FrontEnd/Nodejs/Libraries/Prisma","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/FrontEnd/Nodejs/Libraries/Prisma.mdx","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"frontEnd","previous":{"title":"\u5e38\u7528\u5e93","permalink":"/notes3/docs/category/\u5e38\u7528\u5e93-4"},"next":{"title":"\u5e38\u7528\u5e93","permalink":"/notes3/docs/category/\u5e38\u7528\u5e93-5"}}');var l=s(6070),t=s(5658),n=s(4202),o=s(5048),i=s(1088);const c={sidebar_position:1},p=void 0,u={},d=[{value:"\u5b89\u88c5\u548c\u521d\u59cb\u5316",id:"\u5b89\u88c5\u548c\u521d\u59cb\u5316",level:2},{value:"\u5b89\u88c5\u4f9d\u8d56",id:"\u5b89\u88c5\u4f9d\u8d56",level:3},{value:"\u521d\u59cb\u5316",id:"\u521d\u59cb\u5316",level:3},{value:"\u914d\u7f6e\u6570\u636e\u5e93\u8fde\u63a5",id:"\u914d\u7f6e\u6570\u636e\u5e93\u8fde\u63a5",level:3},{value:"\u6a21\u578b\u5b9a\u4e49\u548c\u751f\u6210",id:"\u6a21\u578b\u5b9a\u4e49\u548c\u751f\u6210",level:2},{value:"\u6a21\u578b\u5b9a\u4e49",id:"\u6a21\u578b\u5b9a\u4e49",level:3},{value:"\u751f\u6210\u5ba2\u6237\u7aef\u4ee3\u7801",id:"\u751f\u6210\u5ba2\u6237\u7aef\u4ee3\u7801",level:3},{value:"\u67b6\u6784\u8fc1\u79fb",id:"\u67b6\u6784\u8fc1\u79fb",level:2},{value:"\u751f\u6210\u6ce8\u91ca",id:"\u751f\u6210\u6ce8\u91ca",level:3}];function h(e){const r={a:"a",code:"code",h2:"h2",h3:"h3",p:"p",pre:"pre",...(0,t.R)(),...e.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsxs)(r.p,{children:[(0,l.jsx)(r.a,{href:"https://www.prisma.io/docs/orm/overview/introduction/what-is-prisma",children:"Prisma"}),"\u662f\u4e00\u4e2a\u5f00\u6e90\u7684",(0,l.jsx)(r.code,{children:"ORM"}),"\u6846\u67b6,\u7528\u4e8e\u4e0e\u6570\u636e\u5e93\u8fdb\u884c\u4ea4\u4e92.\u5b83\u63d0\u4f9b\u4e86\u4e00\u79cd\u7b80\u5355\u800c\u5f3a\u5927\u7684\u65b9\u5f0f\u6765\u7ba1\u7406\u6570\u636e\u5e93\u8fde\u63a5\u3001\u6267\u884c\u67e5\u8be2\u548c\u5904\u7406\u6570\u636e."]}),"\n",(0,l.jsx)(r.h2,{id:"\u5b89\u88c5\u548c\u521d\u59cb\u5316",children:"\u5b89\u88c5\u548c\u521d\u59cb\u5316"}),"\n",(0,l.jsx)(r.h3,{id:"\u5b89\u88c5\u4f9d\u8d56",children:"\u5b89\u88c5\u4f9d\u8d56"}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-bash",children:"# \u63d0\u4f9bcli\u5de5\u5177,\u5f00\u53d1\u65f6\u9700\u8981\u7528\u5230\r\nnpm install prisma --save-dev\r\n\r\n# \u63d0\u4f9b\u6570\u636e\u5e93\u4ea4\u4e92API,\u751f\u4ea7\u65f6\u9700\u8981\u7528\u5230\r\nnpm install @prisma/client\n"})}),"\n",(0,l.jsx)(r.h3,{id:"\u521d\u59cb\u5316",children:"\u521d\u59cb\u5316"}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-bash",children:"npx prisma init\n"})}),"\n",(0,l.jsxs)(r.p,{children:["\u5f53\u6267\u884c\u6b64\u547d\u4ee4\u65f6,",(0,l.jsx)(r.code,{children:"Prisma"}),"\u4f1a\u5728\u9879\u76ee\u6839\u76ee\u5f55\u4e0b\u521b\u5efa\u4e00\u4e2a",(0,l.jsx)(r.code,{children:"prisma"}),"\u76ee\u5f55,\u5176\u4e2d\u5305\u542b\u4e00\u4e2a",(0,l.jsx)(r.code,{children:"schema.prisma"}),"\u6587\u4ef6,\u7528\u4e8e\u5b9a\u4e49\u6570\u636e\u5e93\u6a21\u578b\u548c\u8fde\u63a5\u4fe1\u606f."]}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-text",children:"prisma/\r\n  \u2514\u2500\u2500 schema.prisma    # Prisma \u6a21\u578b\u5b9a\u4e49\u6587\u4ef6\r\n.env                   # \u73af\u5883\u53d8\u91cf\u6587\u4ef6\n"})}),"\n",(0,l.jsx)(r.h3,{id:"\u914d\u7f6e\u6570\u636e\u5e93\u8fde\u63a5",children:"\u914d\u7f6e\u6570\u636e\u5e93\u8fde\u63a5"}),"\n","\n",(0,l.jsx)(n._G,{}),"\n",(0,l.jsxs)(r.p,{children:["\u5f53\u4fee\u6539\u4e86",(0,l.jsx)(r.code,{children:"schema.prisma"}),"\u6587\u4ef6\u540e,\u9700\u8981\u6267\u884c\u6b64\u547d\u4ee4\u6765\u751f\u6210\u5bf9\u5e94\u7684",(0,l.jsx)(r.code,{children:"TypeScript"}),"\u7c7b\u578b\u548c\u6570\u636e\u5e93\u4ea4\u4e92\u4ee3\u7801."]}),"\n",(0,l.jsx)(r.h2,{id:"\u6a21\u578b\u5b9a\u4e49\u548c\u751f\u6210",children:"\u6a21\u578b\u5b9a\u4e49\u548c\u751f\u6210"}),"\n",(0,l.jsx)(r.h3,{id:"\u6a21\u578b\u5b9a\u4e49",children:"\u6a21\u578b\u5b9a\u4e49"}),"\n",(0,l.jsxs)(r.p,{children:["\u5728",(0,l.jsx)(r.code,{children:"schema.prisma"}),"\u6587\u4ef6\u4e2d,\u53ef\u4ee5\u4f7f\u7528",(0,l.jsx)(r.code,{children:"model"}),"\u5173\u952e\u5b57\u5b9a\u4e49\u6570\u636e\u5e93\u6a21\u578b.\u6bcf\u4e2a\u6a21\u578b\u5bf9\u5e94\u6570\u636e\u5e93\u4e2d\u7684\u4e00\u5f20\u8868,\u6a21\u578b\u4e2d\u7684\u5b57\u6bb5\u5bf9\u5e94\u8868\u4e2d\u7684\u5217."]}),"\n","\n",(0,l.jsxs)(o.A,{children:[(0,l.jsx)(i.A,{value:"base",label:"\u57fa\u672c",default:!0,children:(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-prisma",children:'    ///\u7528\u6237\u4fe1\u606f\r\n    model User {\r\n        ///\u7528\u6237id\r\n        id        Int      @id @default(autoincrement())\r\n        ///\u7528\u6237\u540d(\u552f\u4e00)\r\n        username  String   @unique\r\n        ///\u7528\u6237\u5bc6\u7801\r\n        password  String\r\n        ///\u624b\u673a\u53f7(\u552f\u4e00)\r\n        phone     String?  @unique\r\n        ///\u90ae\u7bb1(\u552f\u4e00)\r\n        email     String?  @unique\r\n        ///\u521b\u5efa\u65f6\u95f4\r\n        createdAt DateTime @default(now())\r\n        ///\u66f4\u65b0\u65f6\u95f4\r\n        updatedAt DateTime @updatedAt\r\n\r\n        @@map("users")\r\n    }\n'})})}),(0,l.jsx)(i.A,{value:"pnpm",label:"pnpm"})]}),"\n",(0,l.jsx)(r.h3,{id:"\u751f\u6210\u5ba2\u6237\u7aef\u4ee3\u7801",children:"\u751f\u6210\u5ba2\u6237\u7aef\u4ee3\u7801"}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-bash",children:"npx prisma generate\n"})}),"\n",(0,l.jsxs)(r.p,{children:["\u5f53\u6267\u884c\u6b64\u547d\u4ee4\u65f6,",(0,l.jsx)(r.code,{children:"Prisma"}),"\u4f1a\u5728",(0,l.jsx)(r.code,{children:"node_modules/.prisma/client/"}),"\u76ee\u5f55\u4e0b\u751f\u6210\u7c7b\u4f3c\u4e0b\u9762\u8fd9\u6837\u7684\u5ba2\u6237\u7aef\u4ee3\u7801:"]}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-text",children:"node_modules/.prisma/client/\r\n\u251c\u2500\u2500 index.js                # \u4e3b\u5165\u53e3\u6587\u4ef6\r\n\u251c\u2500\u2500 index.d.ts             # TypeScript \u7c7b\u578b\u5b9a\u4e49\r\n\u251c\u2500\u2500 schema.prisma          # schema \u526f\u672c\r\n\u2514\u2500\u2500 libquery_engine-*.dll  # \u67e5\u8be2\u5f15\u64ce\n"})}),"\n",(0,l.jsxs)(r.p,{children:["\u53c2\u8003",(0,l.jsx)(r.a,{href:"https://poe.com/s/vbPzPtG9lcLPLE1duu0s",children:"\u5f53\u6211\u6267\u884cnpx prisma generate\u65f6\u4f1a\u53d1\u751f\u4ec0\u4e48"})]}),"\n",(0,l.jsx)(r.h2,{id:"\u67b6\u6784\u8fc1\u79fb",children:"\u67b6\u6784\u8fc1\u79fb"}),"\n",(0,l.jsxs)(r.p,{children:[(0,l.jsx)(r.code,{children:"\u67b6\u6784\u8fc1\u79fb"}),"\u662f\u6307\u5728\u6570\u636e\u5e93\u4e2d\u521b\u5efa\u3001\u4fee\u6539\u6216\u5220\u9664\u8868\u3001\u5217\u3001\u7ea6\u675f\u7b49\u6570\u636e\u5e93\u5bf9\u8c61\u7684\u8fc7\u7a0b.",(0,l.jsx)(r.code,{children:"Prisma"}),"\u63d0\u4f9b\u4e86",(0,l.jsx)(r.code,{children:"Prisma Migrate"}),"\u547d\u4ee4\u884c\u5de5\u5177\u6765\u5b9e\u73b0\u8fd9\u4e00\u529f\u80fd."]}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-bash",children:"npx prisma migrate dev --name init\n"})}),"\n",(0,l.jsxs)(r.p,{children:["\u8fd9\u4e2a\u547d\u4ee4\u4f1a\u57fa\u4e8e",(0,l.jsx)(r.code,{children:"schema.prisma"}),"\u6587\u4ef6\u4e2d\u7684\u6a21\u578b\u5b9a\u4e49,\u5728\u6570\u636e\u5e93\u4e2d\u521b\u5efa\u76f8\u5e94\u7684\u8868\u548c\u5217.\u5e76\u5c06\u751f\u6210\u7684\u8fc1\u79fb\u811a\u672c\u4fdd\u5b58\u5230",(0,l.jsx)(r.code,{children:"prisma/migrations"}),"\u76ee\u5f55\u4e0b."]}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-text",children:'\u4f7f\u7528\u6570\u636e\u5e93\u8fc1\u79fb\u66f4\u65b0\u6570\u636e\u5e93\u67b6\u6784\r\n\r\n\u7528\u6cd5\uff1a\r\n  $ prisma migrate [\u547d\u4ee4] [\u9009\u9879]\r\n\r\n\u5f00\u53d1\u73af\u5883\u547d\u4ee4\uff1a\r\n\r\n         dev   \u6839\u636e Prisma schema \u7684\u53d8\u66f4\u521b\u5efa\u8fc1\u79fb\uff0c\u5c06\u5176\u5e94\u7528\u5230\u6570\u636e\u5e93\uff0c\r\n               \u5e76\u89e6\u53d1\u751f\u6210\u5668\uff08\u5982 Prisma Client\uff09\r\n       reset   \u91cd\u7f6e\u6570\u636e\u5e93\u5e76\u5e94\u7528\u6240\u6709\u8fc1\u79fb\uff08\u6240\u6709\u6570\u636e\u5c06\u4e22\u5931\uff09\r\n\r\n\u751f\u4ea7/\u9884\u53d1\u73af\u5883\u547d\u4ee4\uff1a\r\n\r\n      deploy   \u5c06\u5f85\u5904\u7406\u7684\u8fc1\u79fb\u5e94\u7528\u5230\u6570\u636e\u5e93\r\n      status   \u68c0\u67e5\u6570\u636e\u5e93\u8fc1\u79fb\u7684\u72b6\u6001\r\n     resolve   \u89e3\u51b3\u6570\u636e\u5e93\u8fc1\u79fb\u95ee\u9898\uff08\u5982\u57fa\u7ebf\u3001\u5931\u8d25\u7684\u8fc1\u79fb\u3001\u70ed\u4fee\u590d\u7b49\uff09\r\n\r\n\u9002\u7528\u4e8e\u6240\u6709\u73af\u5883\u7684\u547d\u4ee4\uff1a\r\n\r\n        diff   \u6bd4\u8f83\u4e24\u4e2a\u4efb\u610f\u6765\u6e90\u7684\u6570\u636e\u5e93\u67b6\u6784\r\n\r\n\u9009\u9879\uff1a\r\n\r\n  -h, --help   \u663e\u793a\u6b64\u5e2e\u52a9\u4fe1\u606f\r\n    --schema   \u6307\u5b9a Prisma schema \u6587\u4ef6\u7684\u81ea\u5b9a\u4e49\u8def\u5f84\r\n\r\n\u793a\u4f8b\uff1a\r\n\r\n  \u6839\u636e Prisma schema \u7684\u53d8\u66f4\u521b\u5efa\u8fc1\u79fb\uff0c\u5e94\u7528\u5230\u6570\u636e\u5e93\uff0c\u5e76\u89e6\u53d1\u751f\u6210\u5668\r\n  $ prisma migrate dev\r\n\r\n  \u91cd\u7f6e\u6570\u636e\u5e93\u5e76\u5e94\u7528\u6240\u6709\u8fc1\u79fb\r\n  $ prisma migrate reset\r\n\r\n  \u5728\u751f\u4ea7/\u9884\u53d1\u73af\u5883\u4e2d\u5e94\u7528\u5f85\u5904\u7406\u7684\u8fc1\u79fb\r\n  $ prisma migrate deploy\r\n\r\n  \u68c0\u67e5\u751f\u4ea7/\u9884\u53d1\u73af\u5883\u6570\u636e\u5e93\u4e2d\u7684\u8fc1\u79fb\u72b6\u6001\r\n  $ prisma migrate status\r\n\r\n  \u6307\u5b9a schema \u6587\u4ef6\r\n  $ prisma migrate status --schema=./schema.prisma\r\n\r\n  \u6bd4\u8f83\u4e24\u4e2a\u6570\u636e\u5e93\u7684\u67b6\u6784\u5e76\u5c06\u5dee\u5f02\u6e32\u67d3\u4e3a SQL \u811a\u672c\r\n  $ prisma migrate diff \\\r\n    --from-url "$DATABASE_URL" \\\r\n    --to-url "postgresql://login:password@localhost:5432/db" \\\r\n    --script\n'})}),"\n",(0,l.jsx)(r.h3,{id:"\u751f\u6210\u6ce8\u91ca",children:"\u751f\u6210\u6ce8\u91ca"}),"\n",(0,l.jsxs)(r.p,{children:["\u9ed8\u8ba4\u60c5\u51b5\u4e0b,",(0,l.jsx)(r.code,{children:"npx prisma migrate dev"}),"\u547d\u4ee4\u4e0d\u4f1a\u7ed9\u6570\u636e\u5e93\u548c\u5217\u6dfb\u52a0\u6ce8\u91ca.\u5982\u679c\u9700\u8981\u6dfb\u52a0\u6ce8\u91ca,\u9700\u8981\u7528\u5230",(0,l.jsx)(r.a,{href:"https://github.com/onozaty/prisma-db-comments-generator",children:"prisma-db-comments-generator"})]}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-bash",children:"npm install --save-dev @onozaty/prisma-db-comments-generator\n"})}),"\n",(0,l.jsxs)(r.p,{children:["\u7136\u540e\u5728",(0,l.jsx)(r.code,{children:"schema.prisma"}),"\u6587\u4ef6\u4e2d\u4fee\u6539\u751f\u6210\u5668\u914d\u7f6e:"]}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-prisma",children:'generator comments {\r\n  provider = "prisma-db-comments-generator"\r\n}\n'})}),"\n",(0,l.jsxs)(r.p,{children:["\u5728",(0,l.jsx)(r.code,{children:"schema.prisma"}),"\u6587\u4ef6\u4e2d\u4f7f\u7528",(0,l.jsx)(r.code,{children:"///"}),"\u6ce8\u91ca\u6765\u6dfb\u52a0\u6ce8\u91ca,\u4f8b\u5982:"]}),"\n",(0,l.jsx)(r.pre,{children:(0,l.jsx)(r.code,{className:"language-prisma",children:"model User {\r\n    ///\u7528\u6237id\r\n    id        Int      @id @default(autoincrement())\r\n    ///\u7528\u6237\u540d(\u552f\u4e00)\r\n    username  String   @unique\r\n}\n"})}),"\n",(0,l.jsxs)(r.p,{children:["\u6700\u540e\u6267\u884c",(0,l.jsx)(r.code,{children:"npx prisma migrate dev"}),"\u547d\u4ee4,\u5373\u53ef\u5728\u6570\u636e\u5e93\u4e2d\u770b\u5230\u6ce8\u91ca."]})]})}function m(e={}){const{wrapper:r}={...(0,t.R)(),...e.components};return r?(0,l.jsx)(r,{...e,children:(0,l.jsx)(h,{...e})}):h(e)}},1088:(e,r,s)=>{s.d(r,{A:()=>n});s(758);var a=s(3526);const l={tabItem:"tabItem_nvWs"};var t=s(6070);function n(e){let{children:r,hidden:s,className:n}=e;return(0,t.jsx)("div",{role:"tabpanel",className:(0,a.A)(l.tabItem,n),hidden:s,children:r})}},5048:(e,r,s)=>{s.d(r,{A:()=>k});var a=s(758),l=s(3526),t=s(2973),n=s(5557),o=s(7636),i=s(2310),c=s(4919),p=s(1231);function u(e){return a.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,a.isValidElement)(e)&&function(e){const{props:r}=e;return!!r&&"object"==typeof r&&"value"in r}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function d(e){const{values:r,children:s}=e;return(0,a.useMemo)((()=>{const e=r??function(e){return u(e).map((e=>{let{props:{value:r,label:s,attributes:a,default:l}}=e;return{value:r,label:s,attributes:a,default:l}}))}(s);return function(e){const r=(0,c.XI)(e,((e,r)=>e.value===r.value));if(r.length>0)throw new Error(`Docusaurus error: Duplicate values "${r.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[r,s])}function h(e){let{value:r,tabValues:s}=e;return s.some((e=>e.value===r))}function m(e){let{queryString:r=!1,groupId:s}=e;const l=(0,n.W6)(),t=function(e){let{queryString:r=!1,groupId:s}=e;if("string"==typeof r)return r;if(!1===r)return null;if(!0===r&&!s)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return s??null}({queryString:r,groupId:s});return[(0,i.aZ)(t),(0,a.useCallback)((e=>{if(!t)return;const r=new URLSearchParams(l.location.search);r.set(t,e),l.replace({...l.location,search:r.toString()})}),[t,l])]}function b(e){const{defaultValue:r,queryString:s=!1,groupId:l}=e,t=d(e),[n,i]=(0,a.useState)((()=>function(e){let{defaultValue:r,tabValues:s}=e;if(0===s.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(r){if(!h({value:r,tabValues:s}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${r}" but none of its children has the corresponding value. Available values are: ${s.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return r}const a=s.find((e=>e.default))??s[0];if(!a)throw new Error("Unexpected error: 0 tabValues");return a.value}({defaultValue:r,tabValues:t}))),[c,u]=m({queryString:s,groupId:l}),[b,x]=function(e){let{groupId:r}=e;const s=function(e){return e?`docusaurus.tab.${e}`:null}(r),[l,t]=(0,p.Dv)(s);return[l,(0,a.useCallback)((e=>{s&&t.set(e)}),[s,t])]}({groupId:l}),g=(()=>{const e=c??b;return h({value:e,tabValues:t})?e:null})();(0,o.A)((()=>{g&&i(g)}),[g]);return{selectedValue:n,selectValue:(0,a.useCallback)((e=>{if(!h({value:e,tabValues:t}))throw new Error(`Can't select invalid tab value=${e}`);i(e),u(e),x(e)}),[u,x,t]),tabValues:t}}var x=s(1760);const g={tabList:"tabList_vBCw",tabItem:"tabItem_NxBH"};var j=s(6070);function v(e){let{className:r,block:s,selectedValue:a,selectValue:n,tabValues:o}=e;const i=[],{blockElementScrollPositionUntilNextRender:c}=(0,t.a_)(),p=e=>{const r=e.currentTarget,s=i.indexOf(r),l=o[s].value;l!==a&&(c(r),n(l))},u=e=>{let r=null;switch(e.key){case"Enter":p(e);break;case"ArrowRight":{const s=i.indexOf(e.currentTarget)+1;r=i[s]??i[0];break}case"ArrowLeft":{const s=i.indexOf(e.currentTarget)-1;r=i[s]??i[i.length-1];break}}r?.focus()};return(0,j.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,l.A)("tabs",{"tabs--block":s},r),children:o.map((e=>{let{value:r,label:s,attributes:t}=e;return(0,j.jsx)("li",{role:"tab",tabIndex:a===r?0:-1,"aria-selected":a===r,ref:e=>i.push(e),onKeyDown:u,onClick:p,...t,className:(0,l.A)("tabs__item",g.tabItem,t?.className,{"tabs__item--active":a===r}),children:s??r},r)}))})}function f(e){let{lazy:r,children:s,selectedValue:t}=e;const n=(Array.isArray(s)?s:[s]).filter(Boolean);if(r){const e=n.find((e=>e.props.value===t));return e?(0,a.cloneElement)(e,{className:(0,l.A)("margin-top--md",e.props.className)}):null}return(0,j.jsx)("div",{className:"margin-top--md",children:n.map(((e,r)=>(0,a.cloneElement)(e,{key:r,hidden:e.props.value!==t})))})}function y(e){const r=b(e);return(0,j.jsxs)("div",{className:(0,l.A)("tabs-container",g.tabList),children:[(0,j.jsx)(v,{...r,...e}),(0,j.jsx)(f,{...r,...e})]})}function k(e){const r=(0,x.A)();return(0,j.jsx)(y,{...e,children:u(e.children)},String(r))}},4202:(e,r,s)=>{s.d(r,{ey:()=>w,_G:()=>$,Zp:()=>y,oA:()=>I,ZH:()=>S});var a=s(758),l=s(2944),t=s(1190),n=s(2432),o=s(2568),i=s(8415),c=s(2759),p=s(4896),u=s(3738),d=s(1073),h=s(1546),m=s(9490),b=s(2740),x=s(2045),g=s(4388),j=s(2884),v=s(6070);const f=[{value:"build",label:"build - \u6784\u5efa\u6216\u91cd\u65b0\u6784\u5efa\u670d\u52a1"},{value:"config",label:"config - \u9a8c\u8bc1\u5e76\u67e5\u770b Compose \u6587\u4ef6"},{value:"create",label:"create - \u521b\u5efa\u670d\u52a1"},{value:"down",label:"down - \u505c\u6b62\u5e76\u79fb\u9664\u8d44\u6e90"},{value:"events",label:"events - \u63a5\u6536\u5bb9\u5668\u5b9e\u65f6\u4e8b\u4ef6"},{value:"exec",label:"exec - \u5728\u8fd0\u884c\u4e2d\u7684\u5bb9\u5668\u4e2d\u6267\u884c\u547d\u4ee4"},{value:"images",label:"images - \u5217\u51fa\u955c\u50cf"},{value:"kill",label:"kill - \u6740\u6b7b\u5bb9\u5668"},{value:"logs",label:"logs - \u67e5\u770b\u5bb9\u5668\u8f93\u51fa"},{value:"pause",label:"pause - \u6682\u505c\u670d\u52a1"},{value:"port",label:"port - \u6253\u5370\u7aef\u53e3\u6620\u5c04"},{value:"ps",label:"ps - \u5217\u51fa\u5bb9\u5668"},{value:"pull",label:"pull - \u62c9\u53d6\u670d\u52a1\u955c\u50cf"},{value:"push",label:"push - \u63a8\u9001\u670d\u52a1\u955c\u50cf"},{value:"restart",label:"restart - \u91cd\u542f\u670d\u52a1"},{value:"rm",label:"rm - \u79fb\u9664\u505c\u6b62\u7684\u5bb9\u5668"},{value:"run",label:"run - \u8fd0\u884c\u4e00\u6b21\u6027\u547d\u4ee4"},{value:"scale",label:"scale - \u8bbe\u7f6e\u670d\u52a1\u7684\u5bb9\u5668\u6570\u91cf"},{value:"start",label:"start - \u542f\u52a8\u670d\u52a1"},{value:"stop",label:"stop - \u505c\u6b62\u670d\u52a1"},{value:"top",label:"top - \u663e\u793a\u8fd0\u884c\u4e2d\u7684\u8fdb\u7a0b"},{value:"unpause",label:"unpause - \u53d6\u6d88\u6682\u505c\u670d\u52a1"},{value:"up",label:"up - \u521b\u5efa\u5e76\u542f\u52a8\u5bb9\u5668"},{value:"version",label:"version - \u663e\u793a\u7248\u672c\u4fe1\u606f"}];function y(){const[e,r]=(0,a.useState)(""),[s,y]=(0,a.useState)({}),k=(0,b.m)({initialValues:{command:"up",services:[],detach:!0,build:!1,noDeps:!1,forceRecreate:!1,removeOrphans:!1,timeout:10,volumes:!1,removeImages:"",follow:!1,tail:0,timestamps:!1,user:"",workdir:"",entrypoint:"",scale:{}}});return(0,v.jsx)(c.Z,{shadow:"sm",p:"lg",children:(0,v.jsx)("form",{onSubmit:k.onSubmit((e=>{const s=(e=>{const r=["docker compose"];switch(e.file&&r.push(`-f ${e.file}`),e.projectName&&r.push(`-p ${e.projectName}`),e.profile&&r.push(`--profile ${e.profile}`),r.push(e.command),e.command){case"up":e.detach&&r.push("-d"),e.build&&r.push("--build"),e.noDeps&&r.push("--no-deps"),e.forceRecreate&&r.push("--force-recreate"),e.removeOrphans&&r.push("--remove-orphans"),10!==e.timeout&&r.push(`--timeout ${e.timeout}`);break;case"down":e.volumes&&r.push("-v"),e.removeImages&&r.push(`--rmi ${e.removeImages}`),e.removeOrphans&&r.push("--remove-orphans");break;case"logs":e.follow&&r.push("-f"),e.timestamps&&r.push("-t"),void 0!==e.tail&&e.tail>=0&&r.push(`--tail=${e.tail}`);break;case"run":case"exec":e.user&&r.push(`--user ${e.user}`),e.workdir&&r.push(`--workdir ${e.workdir}`),e.entrypoint&&r.push(`--entrypoint ${e.entrypoint}`);break;case"scale":const s=Object.entries(e.scale||{}).map((e=>{let[r,s]=e;return`${r}=${s}`}));s.length>0&&r.push(s.join(" "))}return["up","run","exec","logs","rm","start","stop","restart"].includes(e.command)&&e.services.length>0&&r.push(e.services.join(" ")),r.join(" ")})(e);r(s),navigator.clipboard.writeText(s).then((()=>x.$e.show({message:"\u547d\u4ee4\u5df2\u590d\u5236\u5230\u526a\u8d34\u677f",color:"green",position:"top-center"}))).catch((()=>x.$e.show({message:"\u590d\u5236\u5931\u8d25",color:"red",position:"top-center"})))})),children:(0,v.jsxs)(i.B,{gap:"md",children:[(0,v.jsx)(n.l,{label:"Docker Compose \u547d\u4ee4",data:f,searchable:!0,...k.getInputProps("command")}),(0,v.jsx)(o.k,{label:"Compose \u6587\u4ef6\u8def\u5f84",placeholder:"docker-compose.yml",...k.getInputProps("file")}),(0,v.jsx)(o.k,{label:"\u9879\u76ee\u540d\u79f0",placeholder:"my-project",...k.getInputProps("projectName")}),(0,v.jsx)(o.k,{label:"Profile",placeholder:"\u5f00\u53d1\u73af\u5883\u3001\u751f\u4ea7\u73af\u5883\u7b49",...k.getInputProps("profile")}),["up","run","exec","logs","rm","start","stop","restart","scale"].includes(k.values.command)&&(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(n.l,{label:"\u6dfb\u52a0\u670d\u52a1",placeholder:"\u9009\u62e9\u8981\u64cd\u4f5c\u7684\u670d\u52a1",data:[{value:"web",label:"Web \u670d\u52a1"},{value:"db",label:"\u6570\u636e\u5e93"},{value:"redis",label:"Redis"},{value:"nginx",label:"Nginx"}],onChange:e=>{return e&&!(!(r=e)||k.values.services.includes(r)||(k.setFieldValue("services",[...k.values.services,r]),0));var r},clearable:!0}),k.values.services.length>0&&(0,v.jsxs)(i.B,{gap:"xs",children:[(0,v.jsx)(p.E,{size:"sm",fw:500,children:"\u5df2\u9009\u62e9\u7684\u670d\u52a1\uff1a"}),(0,v.jsx)(u.Y,{gap:"xs",children:k.values.services.map((e=>(0,v.jsx)(d.E,{size:"lg",rightSection:(0,v.jsx)(h.M,{size:"xs",color:"red",variant:"transparent",onClick:()=>(e=>{k.setFieldValue("services",k.values.services.filter((r=>r!==e)))})(e),children:(0,v.jsx)(j.qbC,{size:10})}),children:e},e)))})]})]}),(()=>{switch(k.values.command){case"up":return(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(l.d,{label:"\u540e\u53f0\u8fd0\u884c (-d)",...k.getInputProps("detach",{type:"checkbox"})}),(0,v.jsx)(l.d,{label:"\u6784\u5efa\u955c\u50cf (--build)",...k.getInputProps("build",{type:"checkbox"})}),(0,v.jsx)(l.d,{label:"\u4e0d\u542f\u52a8\u4f9d\u8d56 (--no-deps)",...k.getInputProps("noDeps",{type:"checkbox"})}),(0,v.jsx)(l.d,{label:"\u5f3a\u5236\u91cd\u65b0\u521b\u5efa (--force-recreate)",...k.getInputProps("forceRecreate",{type:"checkbox"})}),(0,v.jsx)(l.d,{label:"\u79fb\u9664\u5b64\u7acb\u5bb9\u5668 (--remove-orphans)",...k.getInputProps("removeOrphans",{type:"checkbox"})}),(0,v.jsx)(t.Q,{label:"\u8d85\u65f6\u65f6\u95f4\uff08\u79d2\uff09",...k.getInputProps("timeout"),min:0})]});case"down":return(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(l.d,{label:"\u79fb\u9664\u5377 (-v)",...k.getInputProps("volumes",{type:"checkbox"})}),(0,v.jsx)(n.l,{label:"\u79fb\u9664\u955c\u50cf",placeholder:"\u9009\u62e9\u8981\u79fb\u9664\u7684\u955c\u50cf\u7c7b\u578b",data:[{value:"",label:"\u4e0d\u79fb\u9664"},{value:"local",label:"\u4ec5\u672c\u5730\u955c\u50cf"},{value:"all",label:"\u6240\u6709\u955c\u50cf"}],...k.getInputProps("removeImages")})]});case"logs":return(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(l.d,{label:"\u8ddf\u8e2a\u65e5\u5fd7 (-f)",...k.getInputProps("follow",{type:"checkbox"})}),(0,v.jsx)(l.d,{label:"\u663e\u793a\u65f6\u95f4\u6233 (-t)",...k.getInputProps("timestamps",{type:"checkbox"})}),(0,v.jsx)(t.Q,{label:"\u663e\u793a\u6700\u540e\u51e0\u884c",...k.getInputProps("tail"),min:0})]});case"run":case"exec":return(0,v.jsxs)(v.Fragment,{children:[(0,v.jsx)(o.k,{label:"\u7528\u6237",placeholder:"username:group",...k.getInputProps("user")}),(0,v.jsx)(o.k,{label:"\u5de5\u4f5c\u76ee\u5f55",placeholder:"/app",...k.getInputProps("workdir")}),(0,v.jsx)(o.k,{label:"\u5165\u53e3\u70b9",placeholder:"custom-entrypoint",...k.getInputProps("entrypoint")})]});case"scale":return(0,v.jsx)(i.B,{gap:"xs",children:k.values.services.map((e=>(0,v.jsx)(t.Q,{label:`${e} \u5bb9\u5668\u6570\u91cf`,min:0,value:s[e]||0,onChange:r=>{const a={...s,[e]:r||0};y(a),k.setFieldValue("scale",a)}},e)))});default:return null}})(),e&&(0,v.jsxs)(c.Z,{withBorder:!0,children:[(0,v.jsx)(p.E,{size:"sm",fw:500,children:"\u751f\u6210\u7684\u547d\u4ee4\uff1a"}),(0,v.jsx)(p.E,{style:{wordBreak:"break-all"},mt:"xs",children:e})]}),(0,v.jsx)(m.$,{type:"submit",leftSection:(0,v.jsx)(g.y9h,{size:14}),children:"\u751f\u6210\u547d\u4ee4\u5e76\u590d\u5236"})]})})})}var k=s(9178),P=s(3121);function I(){const e=(0,b.m)({initialValues:{url:"",aria2:!1,aria2Addr:"localhost:6800",aria2Method:"http",audioOnly:!1,caption:!1,chunkSize:1,debug:!1,end:0,episodeTitleOnly:!1,fileNameLength:255,info:!1,json:!1,multiThread:!1,playlist:!1,retry:10,silent:!1,start:1,threadNum:10,youkuCcode:"0502"}});return(0,v.jsx)(c.Z,{shadow:"sm",padding:"lg",children:(0,v.jsx)("form",{onSubmit:e.onSubmit((e=>{const r=(e=>{const r=["lux"],s=!1,a="localhost:6800",l="http";return e.aria2&&e.aria2!==s&&r.push("--aria2"),e.aria2&&(e.aria2Addr&&e.aria2Addr!==a&&r.push(`--aria2-addr "${e.aria2Addr}"`),e.aria2Method&&e.aria2Method!==l&&r.push(`--aria2-method "${e.aria2Method}"`),e.aria2Token&&r.push(`--aria2-token "${e.aria2Token}"`)),r.push(e.url),r.join(" ")})(e);navigator.clipboard.writeText(r).then((()=>x.$e.show({message:"\u547d\u4ee4\u5df2\u590d\u5236\u5230\u526a\u8d34\u677f",color:"green",position:"top-center"}))).catch((()=>x.$e.show({message:"\u590d\u5236\u5931\u8d25",color:"red",position:"top-center"})))})),children:(0,v.jsxs)(i.B,{children:[(0,v.jsx)(o.k,{required:!0,label:"\u89c6\u9891URL",placeholder:"\u8bf7\u8f93\u5165\u8981\u4e0b\u8f7d\u7684\u89c6\u9891URL",...e.getInputProps("url")}),(0,v.jsx)(l.d,{label:"Aria2\u4e0b\u8f7d",...e.getInputProps("aria2",{type:"checkbox"})}),(0,v.jsx)(o.k,{label:"Aria2\u5730\u5740",placeholder:"localhost:6800",...e.getInputProps("aria2Addr")}),(0,v.jsx)(n.l,{label:"Aria2\u65b9\u6cd5",data:[{value:"http",label:"HTTP"},{value:"https",label:"HTTPS"}],...e.getInputProps("aria2Method")}),(0,v.jsx)(o.k,{label:"Aria2\u4ee4\u724c",placeholder:"Aria2 RPC\u4ee4\u724c",...e.getInputProps("aria2Token")}),(0,v.jsx)(k.T,{label:"Cookie",placeholder:"Cookie",...e.getInputProps("cookie")}),(0,v.jsx)(l.d,{label:"\u8c03\u8bd5\u6a21\u5f0f",...e.getInputProps("debug",{type:"checkbox"})}),(0,v.jsx)(t.Q,{label:"\u7ed3\u675f\u9879",min:0,...e.getInputProps("end")}),(0,v.jsx)(l.d,{label:"\u7eaf\u96c6\u6807\u9898",...e.getInputProps("episodeTitleOnly",{type:"checkbox"})}),(0,v.jsx)(o.k,{label:"URL\u6587\u4ef6",placeholder:"URL\u6587\u4ef6\u8def\u5f84",...e.getInputProps("file")}),(0,v.jsx)(t.Q,{label:"\u6587\u4ef6\u540d\u957f\u5ea6\u9650\u5236",min:0,...e.getInputProps("fileNameLength")}),(0,v.jsx)(l.d,{label:"\u4ec5\u663e\u793a\u4fe1\u606f",...e.getInputProps("info",{type:"checkbox"})}),(0,v.jsx)(o.k,{label:"\u6307\u5b9a\u9879\u76ee",placeholder:"\u5982: 1,5,6,8-10",...e.getInputProps("items")}),(0,v.jsx)(l.d,{label:"\u8f93\u51faJSON",...e.getInputProps("json",{type:"checkbox"})}),(0,v.jsx)(o.k,{label:"Referer",...e.getInputProps("refer")}),(0,v.jsx)(l.d,{label:"\u9759\u9ed8\u6a21\u5f0f",...e.getInputProps("silent",{type:"checkbox"})}),(0,v.jsx)(t.Q,{label:"\u8d77\u59cb\u9879",min:1,...e.getInputProps("start")}),(0,v.jsx)(o.k,{label:"\u6d41\u683c\u5f0f",...e.getInputProps("streamFormat")}),(0,v.jsx)(o.k,{label:"User-Agent",...e.getInputProps("userAgent")}),(0,v.jsx)(o.k,{label:"\u4f18\u9177ccode",...e.getInputProps("youkuCcode")}),(0,v.jsx)(o.k,{label:"\u4f18\u9177ckey",...e.getInputProps("youkuCkey")}),(0,v.jsx)(P.y,{label:"\u4f18\u9177\u5bc6\u7801",...e.getInputProps("youkuPassword")}),(0,v.jsx)(m.$,{type:"submit",leftSection:(0,v.jsx)(g.y9h,{size:14}),children:"\u751f\u6210\u547d\u4ee4\u5e76\u590d\u5236"})]})})})}function w(){const[e,r]=(0,a.useState)(""),[s,t]=(0,a.useState)(""),f=(0,b.m)({initialValues:{domains:[],challengeType:"dns-01",staging:!1,agreeTos:!0,quiet:!1,nonInteractive:!1},validate:{domains:e=>0===e.length?"\u8bf7\u81f3\u5c11\u6dfb\u52a0\u4e00\u4e2a\u57df\u540d":null,email:e=>e?/^\S+@\S+$/.test(e)?null:"\u8bf7\u8f93\u5165\u6709\u6548\u7684\u90ae\u7bb1\u5730\u5740":null}}),y=()=>{e&&!f.values.domains.includes(e)&&(f.setFieldValue("domains",[...f.values.domains,e]),r(""))};return(0,v.jsx)(c.Z,{shadow:"sm",p:"lg",children:(0,v.jsx)("form",{onSubmit:f.onSubmit((e=>{const r=(e=>{const r=["certbot","certonly"];return e.domains.forEach((e=>{r.push(`-d ${e}`)})),r.push("--manual"),r.push(`--preferred-challenges ${e.challengeType}`),e.staging&&r.push("--staging"),e.agreeTos&&r.push("--agree-tos"),e.email&&r.push(`--email ${e.email}`),e.serverType&&r.push(`--server-type ${e.serverType}`),e.certPath&&r.push(`--cert-path ${e.certPath}`),e.keyPath&&r.push(`--key-path ${e.keyPath}`),e.quiet&&r.push("--quiet"),e.nonInteractive&&r.push("--non-interactive"),r.join(" ")})(e);t(r),navigator.clipboard.writeText(r).then((()=>x.$e.show({message:"\u547d\u4ee4\u5df2\u590d\u5236\u5230\u526a\u8d34\u677f",color:"green",position:"top-center"}))).catch((()=>x.$e.show({message:"\u590d\u5236\u5931\u8d25",color:"red",position:"top-center"})))})),children:(0,v.jsxs)(i.B,{gap:"md",children:[(0,v.jsx)(u.Y,{align:"flex-start",children:(0,v.jsx)(o.k,{style:{flex:1},label:"\u6dfb\u52a0\u57df\u540d",placeholder:"\u8f93\u5165\u57df\u540d\u540e\u6309\u56de\u8f66\u6216\u70b9\u51fb\u6dfb\u52a0\u6309\u94ae",value:e,onChange:e=>r(e.currentTarget.value),onKeyPress:e=>{"Enter"===e.key&&(e.preventDefault(),y())},rightSection:(0,v.jsx)(h.M,{onClick:y,disabled:!e,variant:"filled",color:"blue",children:(0,v.jsx)(j.OiG,{size:14})})})}),f.values.domains.length>0&&(0,v.jsxs)(i.B,{gap:"xs",children:[(0,v.jsx)(p.E,{size:"sm",fw:500,children:"\u5df2\u6dfb\u52a0\u7684\u57df\u540d\uff1a"}),(0,v.jsx)(u.Y,{gap:"xs",children:f.values.domains.map((e=>(0,v.jsx)(d.E,{size:"lg",rightSection:(0,v.jsx)(h.M,{size:"xs",color:"red",variant:"transparent",onClick:()=>(e=>{f.setFieldValue("domains",f.values.domains.filter((r=>r!==e)))})(e),children:(0,v.jsx)(j.qbC,{size:10})}),children:e},e)))})]}),(0,v.jsx)(n.l,{label:"\u9a8c\u8bc1\u65b9\u5f0f",data:[{value:"dns-01",label:"DNS \u9a8c\u8bc1"},{value:"http-01",label:"HTTP \u9a8c\u8bc1"},{value:"tls-alpn-01",label:"TLS-ALPN \u9a8c\u8bc1"}],...f.getInputProps("challengeType")}),(0,v.jsx)(o.k,{label:"\u90ae\u7bb1\u5730\u5740",placeholder:"\u7528\u4e8e\u63a5\u6536\u8bc1\u4e66\u8fc7\u671f\u901a\u77e5",...f.getInputProps("email")}),(0,v.jsx)(l.d,{label:"\u6d4b\u8bd5\u6a21\u5f0f",description:"\u4f7f\u7528 Let's Encrypt \u7684\u6d4b\u8bd5\u73af\u5883",...f.getInputProps("staging",{type:"checkbox"})}),(0,v.jsx)(l.d,{label:"\u540c\u610f\u670d\u52a1\u6761\u6b3e",...f.getInputProps("agreeTos",{type:"checkbox"})}),(0,v.jsx)(l.d,{label:"\u5b89\u9759\u6a21\u5f0f",description:"\u51cf\u5c11\u8f93\u51fa\u4fe1\u606f",...f.getInputProps("quiet",{type:"checkbox"})}),(0,v.jsx)(l.d,{label:"\u975e\u4ea4\u4e92\u5f0f",description:"\u4e0d\u9700\u8981\u7528\u6237\u8f93\u5165",...f.getInputProps("nonInteractive",{type:"checkbox"})}),(0,v.jsx)(o.k,{label:"\u8bc1\u4e66\u8def\u5f84",placeholder:"/etc/letsencrypt/live/domain/fullchain.pem",...f.getInputProps("certPath")}),(0,v.jsx)(o.k,{label:"\u79c1\u94a5\u8def\u5f84",placeholder:"/etc/letsencrypt/live/domain/privkey.pem",...f.getInputProps("keyPath")}),s&&(0,v.jsxs)(c.Z,{withBorder:!0,children:[(0,v.jsx)(p.E,{size:"sm",fw:500,children:"\u751f\u6210\u7684\u547d\u4ee4\uff1a"}),(0,v.jsx)(p.E,{style:{wordBreak:"break-all"},mt:"xs",children:s})]}),(0,v.jsx)(m.$,{type:"submit",leftSection:(0,v.jsx)(g.y9h,{size:14}),disabled:0===f.values.domains.length,children:"\u751f\u6210\u547d\u4ee4\u5e76\u590d\u5236"})]})})})}function S(){const e=(0,b.m)({initialValues:{apiEndpoint:"https://api.openai.com/v1/chat/completions",apiKey:"",model:"gpt-4",systemPrompt:"You are a helpful assistant.",userPrompt:"Hello!"},validate:{apiEndpoint:e=>e?null:"\u8bf7\u8f93\u5165 API \u5730\u5740",apiKey:e=>e?null:"\u8bf7\u8f93\u5165 API Key"}}),r=e=>{const r={model:e.model,messages:[{role:"system",content:e.systemPrompt},{role:"user",content:e.userPrompt}]};return`curl ${e.apiEndpoint} \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer ${e.apiKey}" \\\n  -d '${JSON.stringify(r,null,2)}'`};return(0,v.jsx)(c.Z,{shadow:"sm",p:"lg",children:(0,v.jsx)("form",{onSubmit:e.onSubmit((e=>{const s=r(e);navigator.clipboard.writeText(s).then((()=>x.$e.show({message:"\u547d\u4ee4\u5df2\u590d\u5236\u5230\u526a\u8d34\u677f",color:"green",position:"top-center"}))).catch((()=>x.$e.show({message:"\u590d\u5236\u5931\u8d25",color:"red",position:"top-center"})))})),children:(0,v.jsxs)(i.B,{gap:"md",children:[(0,v.jsx)(o.k,{label:"API \u5730\u5740",placeholder:"https://api.openai.com/v1/chat/completions",...e.getInputProps("apiEndpoint")}),(0,v.jsx)(o.k,{label:"API Key",type:"password",placeholder:"sk-...",...e.getInputProps("apiKey")}),(0,v.jsx)(n.l,{label:"\u6a21\u578b",data:[{value:"gpt-4",label:"GPT-4"},{value:"gpt-3.5-turbo",label:"GPT-3.5 Turbo"}],...e.getInputProps("model")}),(0,v.jsx)(o.k,{label:"System Prompt",placeholder:"\u7cfb\u7edf\u63d0\u793a\u8bcd",...e.getInputProps("systemPrompt")}),(0,v.jsx)(o.k,{label:"User Prompt",placeholder:"\u7528\u6237\u63d0\u793a\u8bcd",...e.getInputProps("userPrompt")}),(0,v.jsx)(m.$,{type:"submit",leftSection:(0,v.jsx)(g.y9h,{size:14}),children:"\u751f\u6210\u547d\u4ee4\u5e76\u590d\u5236"}),e.values.apiEndpoint&&e.values.apiKey&&(0,v.jsxs)(c.Z,{withBorder:!0,children:[(0,v.jsx)(p.E,{size:"sm",fw:500,children:"\u751f\u6210\u7684 curl \u547d\u4ee4\uff1a"}),(0,v.jsx)(p.E,{style:{wordBreak:"break-all",whiteSpace:"pre-wrap"},mt:"xs",children:r(e.values)})]})]})})})}function $(){const[e,r]=(0,a.useState)(""),s=(0,b.m)({initialValues:{provider:"postgresql",username:"",password:"",host:"localhost",port:5432,database:"",schema:"public"},validate:{username:e=>e?null:"\u8bf7\u8f93\u5165\u7528\u6237\u540d",password:e=>e?null:"\u8bf7\u8f93\u5165\u5bc6\u7801",host:e=>e?null:"\u8bf7\u8f93\u5165\u4e3b\u673a\u5730\u5740",port:e=>e?null:"\u8bf7\u8f93\u5165\u7aef\u53e3\u53f7",database:e=>e?null:"\u8bf7\u8f93\u5165\u6570\u636e\u5e93\u540d"}});return(0,v.jsx)(c.Z,{shadow:"sm",p:"lg",children:(0,v.jsx)("form",{onSubmit:s.onSubmit((e=>{const s=(e=>{const{provider:r,username:s,password:a,host:l,port:t,database:n,schema:o}=e;let i=`${r}://`;return(s||a)&&(i+=`${encodeURIComponent(s)}:${encodeURIComponent(a)}@`),i+=`${l}:${t}`,i+=`/${n}`,"postgresql"===r&&o&&(i+=`?schema=${o}`),i})(e);r(s),navigator.clipboard.writeText(s).then((()=>x.$e.show({message:"\u8fde\u63a5URL\u5df2\u590d\u5236\u5230\u526a\u8d34\u677f",color:"green",position:"top-center"}))).catch((()=>x.$e.show({message:"\u590d\u5236\u5931\u8d25",color:"red",position:"top-center"})))})),children:(0,v.jsxs)(i.B,{gap:"md",children:[(0,v.jsx)(n.l,{label:"\u6570\u636e\u5e93\u7c7b\u578b",data:[{value:"postgresql",label:"PostgreSQL"},{value:"mysql",label:"MySQL"},{value:"sqlserver",label:"SQL Server"},{value:"mongodb",label:"MongoDB"},{value:"sqlite",label:"SQLite"}],onChange:e=>{s.setFieldValue("provider",e||"postgresql"),s.setFieldValue("port",(e=>{switch(e){case"postgresql":default:return 5432;case"mysql":return 3306;case"sqlserver":return 1433;case"mongodb":return 27017}})(e||"postgresql"))},...s.getInputProps("provider")}),(0,v.jsx)(o.k,{label:"\u7528\u6237\u540d",placeholder:"database_user",...s.getInputProps("username")}),(0,v.jsx)(o.k,{label:"\u5bc6\u7801",type:"password",placeholder:"your_password",...s.getInputProps("password")}),(0,v.jsx)(o.k,{label:"\u4e3b\u673a\u5730\u5740",placeholder:"localhost",...s.getInputProps("host")}),(0,v.jsx)(t.Q,{label:"\u7aef\u53e3",placeholder:"5432",...s.getInputProps("port")}),(0,v.jsx)(o.k,{label:"\u6570\u636e\u5e93\u540d",placeholder:"my_database",...s.getInputProps("database")}),"postgresql"===s.values.provider&&(0,v.jsx)(o.k,{label:"Schema",placeholder:"public",...s.getInputProps("schema")}),e&&(0,v.jsxs)(c.Z,{withBorder:!0,children:[(0,v.jsx)(p.E,{size:"sm",fw:500,children:"\u6570\u636e\u5e93\u8fde\u63a5 URL\uff1a"}),(0,v.jsx)(p.E,{style:{wordBreak:"break-all"},mt:"xs",children:e})]}),(0,v.jsx)(m.$,{type:"submit",leftSection:(0,v.jsx)(g.y9h,{size:14}),children:"\u751f\u6210\u8fde\u63a5 URL \u5e76\u590d\u5236"})]})})})}}}]);