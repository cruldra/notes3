"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[2814],{1651:(n,r,e)=>{e.r(r),e.d(r,{assets:()=>i,contentTitle:()=>a,default:()=>d,frontMatter:()=>l,metadata:()=>t,toc:()=>c});const t=JSON.parse('{"id":"Python/\u679a\u4e3e","title":"\u679a\u4e3e","description":"\u5728Python\u4e2d\u901a\u8fc7\u7ee7\u627fEnum\u7c7b\u6765\u521b\u5efa\u4e00\u4e2a\u81ea\u5b9a\u4e49\u679a\u4e3e\u7c7b\u578b.","source":"@site/docs/Python/\u679a\u4e3e.mdx","sourceDirName":"Python","slug":"/Python/\u679a\u4e3e","permalink":"/notes3/docs/Python/\u679a\u4e3e","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/Python/\u679a\u4e3e.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"python","previous":{"title":"JSAPI","permalink":"/notes3/docs/Python/Wechat/\u652f\u4ed8/JSAPI"},"next":{"title":"\u6a21\u5757\u5316","permalink":"/notes3/docs/Python/\u6a21\u5757\u5316"}}');var o=e(6070),s=e(5658);const l={},a=void 0,i={},c=[{value:"\u57fa\u672c\u793a\u4f8b",id:"\u57fa\u672c\u793a\u4f8b",level:2},{value:"\u81ea\u52a8\u7f16\u53f7",id:"\u81ea\u52a8\u7f16\u53f7",level:2},{value:"\u590d\u6742\u503c\u7684\u679a\u4e3e",id:"\u590d\u6742\u503c\u7684\u679a\u4e3e",level:2},{value:"\u5e26\u65b9\u6cd5\u7684\u679a\u4e3e",id:"\u5e26\u65b9\u6cd5\u7684\u679a\u4e3e",level:2},{value:"\u552f\u4e00\u6027\u68c0\u67e5",id:"\u552f\u4e00\u6027\u68c0\u67e5",level:2},{value:"\u679a\u4e3e\u6210\u5458\u7684\u67e5\u627e",id:"\u679a\u4e3e\u6210\u5458\u7684\u67e5\u627e",level:2},{value:"\u6807\u5fd7\u578b\u679a\u4e3e(Flag)",id:"\u6807\u5fd7\u578b\u679a\u4e3eflag",level:2}];function u(n){const r={a:"a",code:"code",h2:"h2",p:"p",pre:"pre",...(0,s.R)(),...n.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsxs)(r.p,{children:["\u5728",(0,o.jsx)(r.code,{children:"Python"}),"\u4e2d\u901a\u8fc7\u7ee7\u627f",(0,o.jsx)(r.a,{href:"https://docs.python.org/3/library/enum.html",children:"Enum"}),"\u7c7b\u6765\u521b\u5efa\u4e00\u4e2a\u81ea\u5b9a\u4e49\u679a\u4e3e\u7c7b\u578b."]}),"\n",(0,o.jsx)(r.h2,{id:"\u57fa\u672c\u793a\u4f8b",children:"\u57fa\u672c\u793a\u4f8b"}),"\n",(0,o.jsx)(r.pre,{children:(0,o.jsx)(r.code,{className:"language-python",children:"from enum import Enum\r\n\r\nclass Color(Enum):\r\n    RED = 1\r\n    GREEN = 2\r\n    BLUE = 3\r\n\r\n# \u4f7f\u7528\r\ncolor = Color.RED\r\nprint(color)  # Color.RED\r\nprint(color.name)  # 'RED'\r\nprint(color.value)  # 1\r\n\r\n# \u904d\u5386\r\nfor color in Color:\r\n    print(color)  # \u8f93\u51fa\u6240\u6709\u679a\u4e3e\u6210\u5458\n"})}),"\n",(0,o.jsx)(r.h2,{id:"\u81ea\u52a8\u7f16\u53f7",children:"\u81ea\u52a8\u7f16\u53f7"}),"\n",(0,o.jsx)(r.pre,{children:(0,o.jsx)(r.code,{className:"language-python",children:"from enum import Enum, auto\r\n\r\nfrom enum import Enum, auto\r\n\r\nclass Status(Enum):\r\n    PENDING = auto()\r\n    RUNNING = auto()\r\n    FINISHED = auto()\r\n\r\nprint(Status.PENDING.value)  # 1\r\nprint(Status.RUNNING.value)  # 2\r\nprint(Status.FINISHED.value)  # 3\n"})}),"\n",(0,o.jsx)(r.h2,{id:"\u590d\u6742\u503c\u7684\u679a\u4e3e",children:"\u590d\u6742\u503c\u7684\u679a\u4e3e"}),"\n",(0,o.jsx)(r.pre,{children:(0,o.jsx)(r.code,{className:"language-python",children:"class ApiConfig(Enum):\r\n    GITHUB = {\r\n        'url': 'https://api.github.com',\r\n        'token': 'xxx'\r\n    }\r\n    GITLAB = {\r\n        'url': 'https://gitlab.com/api',\r\n        'token': 'yyy'\r\n    }\r\n\r\n# \u4f7f\u7528\r\nconfig = ApiConfig.GITHUB.value['url']\n"})}),"\n",(0,o.jsx)(r.h2,{id:"\u5e26\u65b9\u6cd5\u7684\u679a\u4e3e",children:"\u5e26\u65b9\u6cd5\u7684\u679a\u4e3e"}),"\n",(0,o.jsx)(r.pre,{children:(0,o.jsx)(r.code,{className:"language-python",children:"class Operation(Enum):\r\n    ADD = '+'\r\n    SUBTRACT = '-'\r\n\r\n    def calculate(self, x, y):\r\n        if self is Operation.ADD:\r\n            return x + y\r\n        elif self is Operation.SUBTRACT:\r\n            return x - y\r\n\r\n# \u4f7f\u7528\r\nresult = Operation.ADD.calculate(1, 2)  # 3\n"})}),"\n",(0,o.jsx)(r.h2,{id:"\u552f\u4e00\u6027\u68c0\u67e5",children:"\u552f\u4e00\u6027\u68c0\u67e5"}),"\n",(0,o.jsx)(r.pre,{children:(0,o.jsx)(r.code,{className:"language-python",children:"from enum import Enum, unique\r\n\r\n@unique\r\nclass Status(Enum):\r\n    PENDING = 1\r\n    RUNNING = 1  # \u4f1a\u62a5\u9519\uff0c\u56e0\u4e3a\u503c\u91cd\u590d\n"})}),"\n",(0,o.jsx)(r.h2,{id:"\u679a\u4e3e\u6210\u5458\u7684\u67e5\u627e",children:"\u679a\u4e3e\u6210\u5458\u7684\u67e5\u627e"}),"\n",(0,o.jsx)(r.pre,{children:(0,o.jsx)(r.code,{className:"language-python",children:"class Animal(Enum):\r\n    DOG = 1\r\n    CAT = 2\r\n\r\n# \u901a\u8fc7\u540d\u79f0\u67e5\u627e\r\nanimal = Animal['DOG']  # Animal.DOG\r\n\r\n# \u901a\u8fc7\u503c\u67e5\u627e\r\nanimal = Animal(1)  # Animal.DOG\n"})}),"\n",(0,o.jsx)(r.h2,{id:"\u6807\u5fd7\u578b\u679a\u4e3eflag",children:"\u6807\u5fd7\u578b\u679a\u4e3e(Flag)"}),"\n",(0,o.jsx)(r.pre,{children:(0,o.jsx)(r.code,{className:"language-python",children:"from enum import Flag, auto\r\n\r\nclass Permissions(Flag):\r\n    READ = auto()\r\n    WRITE = auto()\r\n    EXECUTE = auto()\r\n\r\n# \u53ef\u4ee5\u7ec4\u5408\u4f7f\u7528\r\nperm = Permissions.READ | Permissions.WRITE\r\nprint(Permissions.READ in perm)  # True\n"})})]})}function d(n={}){const{wrapper:r}={...(0,s.R)(),...n.components};return r?(0,o.jsx)(r,{...n,children:(0,o.jsx)(u,{...n})}):u(n)}},5658:(n,r,e)=>{e.d(r,{R:()=>l,x:()=>a});var t=e(758);const o={},s=t.createContext(o);function l(n){const r=t.useContext(s);return t.useMemo((function(){return"function"==typeof n?n(r):{...r,...n}}),[r,n])}function a(n){let r;return r=n.disableParentContext?"function"==typeof n.components?n.components(o):n.components||o:l(n.components),t.createElement(s.Provider,{value:r},n.children)}}}]);