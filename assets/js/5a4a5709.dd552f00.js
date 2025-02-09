"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[2157],{3781:(n,e,r)=>{r.r(e),r.d(e,{assets:()=>t,contentTitle:()=>d,default:()=>o,frontMatter:()=>s,metadata:()=>l,toc:()=>c});const l=JSON.parse('{"id":"Python/Libraries/Typer","title":"Typer","description":"Typer\u7528\u4e8e\u6784\u5efa\u547d\u4ee4\u884c\u5e94\u7528\u7a0b\u5e8f.","source":"@site/docs/Python/Libraries/Typer.mdx","sourceDirName":"Python/Libraries","slug":"/Python/Libraries/Typer","permalink":"/notes3/docs/Python/Libraries/Typer","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/Python/Libraries/Typer.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"python","previous":{"title":"Tenacity","permalink":"/notes3/docs/Python/Libraries/Tenacity"},"next":{"title":"\u4efb\u52a1\u8c03\u5ea6","permalink":"/notes3/docs/Python/Libraries/\u4efb\u52a1\u8c03\u5ea6"}}');var a=r(6070),i=r(5658);const s={},d=void 0,t={},c=[{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2},{value:"\u57fa\u7840\u793a\u4f8b",id:"\u57fa\u7840\u793a\u4f8b",level:2},{value:"\u6700\u7b80\u5355\u7684\u793a\u4f8b",id:"\u6700\u7b80\u5355\u7684\u793a\u4f8b",level:3},{value:"\u6dfb\u52a0\u547d\u4ee4\u884c\u53c2\u6570",id:"\u6dfb\u52a0\u547d\u4ee4\u884c\u53c2\u6570",level:3},{value:"\u53c2\u6570\u7c7b\u578b",id:"\u53c2\u6570\u7c7b\u578b",level:2},{value:"\u6587\u6863",id:"\u6587\u6863",level:2},{value:"\u547d\u4ee4(Commands)",id:"\u547d\u4ee4commands",level:2},{value:"\u57fa\u672c\u547d\u4ee4\u793a\u4f8b",id:"\u57fa\u672c\u547d\u4ee4\u793a\u4f8b",level:3},{value:"\u81ea\u52a8\u663e\u793a\u5e2e\u52a9\u4fe1\u606f",id:"\u81ea\u52a8\u663e\u793a\u5e2e\u52a9\u4fe1\u606f",level:3},{value:"\u547d\u4ee4\u6392\u5e8f",id:"\u547d\u4ee4\u6392\u5e8f",level:3}];function p(n){const e={a:"a",code:"code",h2:"h2",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",ul:"ul",...(0,i.R)(),...n.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsxs)(e.p,{children:[(0,a.jsx)(e.a,{href:"https://typer.tiangolo.com/",children:"Typer"}),"\u7528\u4e8e\u6784\u5efa\u547d\u4ee4\u884c\u5e94\u7528\u7a0b\u5e8f."]}),"\n",(0,a.jsx)(e.h2,{id:"\u5b89\u88c5",children:"\u5b89\u88c5"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-bash",children:"pip install typer\n"})}),"\n",(0,a.jsx)(e.h2,{id:"\u57fa\u7840\u793a\u4f8b",children:"\u57fa\u7840\u793a\u4f8b"}),"\n",(0,a.jsx)(e.h3,{id:"\u6700\u7b80\u5355\u7684\u793a\u4f8b",children:"\u6700\u7b80\u5355\u7684\u793a\u4f8b"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'import typer\r\n\r\ndef main():\r\n    print("Hello World")\r\n\r\nif __name__ == "__main__":\r\n    typer.run(main)\n'})}),"\n",(0,a.jsx)(e.p,{children:"\u8fd0\u884c:"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-bash",children:"$ python main.py\r\nHello World\r\n\r\n$ python main.py --help  # \u81ea\u52a8\u751f\u6210\u5e2e\u52a9\u4fe1\u606f\n"})}),"\n",(0,a.jsx)(e.h3,{id:"\u6dfb\u52a0\u547d\u4ee4\u884c\u53c2\u6570",children:"\u6dfb\u52a0\u547d\u4ee4\u884c\u53c2\u6570"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'import typer\r\n\r\ndef main(name: str, lastname: str = "", formal: bool = False):\r\n    """\r\n    \u5411NAME\u6253\u62db\u547c,\u53ef\u4ee5\u9009\u62e9\u6dfb\u52a0--lastname\u3002\r\n    \u4f7f\u7528--formal\u53ef\u4ee5\u66f4\u6b63\u5f0f\u5730\u6253\u62db\u547c\u3002\r\n    """\r\n    if formal:\r\n        print(f"Good day Ms. {name} {lastname}.")\r\n    else:\r\n        print(f"Hello {name} {lastname}")\r\n\r\nif __name__ == "__main__":\r\n    typer.run(main)\n'})}),"\n",(0,a.jsx)(e.p,{children:"\u8fd0\u884c:"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-bash",children:"$ python main.py Camila  # \u5fc5\u9700\u53c2\u6570\r\nHello Camila\r\n\r\n$ python main.py Camila --lastname Guti\xe9rrez  # \u53ef\u9009\u53c2\u6570\r\nHello Camila Guti\xe9rrez\r\n\r\n$ python main.py Camila --lastname Guti\xe9rrez --formal  # \u5e03\u5c14\u6807\u5fd7\r\nGood day Ms. Camila Guti\xe9rrez.\n"})}),"\n",(0,a.jsx)(e.h2,{id:"\u53c2\u6570\u7c7b\u578b",children:"\u53c2\u6570\u7c7b\u578b"}),"\n",(0,a.jsx)(e.p,{children:"Typer\u652f\u6301\u4ee5\u4e0b\u51e0\u79cd\u4e3b\u8981\u7684\u53c2\u6570\u7c7b\u578b:"}),"\n",(0,a.jsxs)(e.ol,{children:["\n",(0,a.jsxs)(e.li,{children:["\n",(0,a.jsx)(e.p,{children:"CLI Arguments (\u4f4d\u7f6e\u53c2\u6570)"}),"\n",(0,a.jsxs)(e.ul,{children:["\n",(0,a.jsx)(e.li,{children:"\u6309\u987a\u5e8f\u4f20\u9012"}),"\n",(0,a.jsx)(e.li,{children:"\u9ed8\u8ba4\u4e3a\u5fc5\u9700\u53c2\u6570"}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(e.li,{children:["\n",(0,a.jsx)(e.p,{children:"CLI Options (\u9009\u9879\u53c2\u6570)"}),"\n",(0,a.jsxs)(e.ul,{children:["\n",(0,a.jsxs)(e.li,{children:["\u4f7f\u7528",(0,a.jsx)(e.code,{children:"--"}),"\u524d\u7f00"]}),"\n",(0,a.jsx)(e.li,{children:"\u9ed8\u8ba4\u4e3a\u53ef\u9009\u53c2\u6570"}),"\n",(0,a.jsx)(e.li,{children:"\u53ef\u4ee5\u5728\u547d\u4ee4\u4e2d\u4efb\u610f\u4f4d\u7f6e\u4f7f\u7528"}),"\n"]}),"\n"]}),"\n",(0,a.jsxs)(e.li,{children:["\n",(0,a.jsx)(e.p,{children:"\u5e03\u5c14\u6807\u5fd7"}),"\n",(0,a.jsxs)(e.ul,{children:["\n",(0,a.jsx)(e.li,{children:"\u4e0d\u9700\u8981\u503c\u7684\u9009\u9879"}),"\n",(0,a.jsxs)(e.li,{children:["\u4f8b\u5982",(0,a.jsx)(e.code,{children:"--formal"})]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,a.jsx)(e.h2,{id:"\u6587\u6863",children:"\u6587\u6863"}),"\n",(0,a.jsx)(e.p,{children:"\u901a\u8fc7\u6dfb\u52a0\u51fd\u6570\u6587\u6863\u5b57\u7b26\u4e32,\u53ef\u4ee5\u81ea\u52a8\u751f\u6210CLI\u5e2e\u52a9\u4fe1\u606f:"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'def main(name: str):\r\n    """\r\n    \u8fd9\u662f\u4e00\u4e2a\u793a\u4f8b\u7a0b\u5e8f\u3002\r\n    \u8fd9\u6bb5\u63cf\u8ff0\u4f1a\u663e\u793a\u5728\u5e2e\u52a9\u4fe1\u606f\u4e2d\u3002\r\n    """\r\n    print(f"Hello {name}")\n'})}),"\n",(0,a.jsxs)(e.p,{children:["\u4f7f\u7528",(0,a.jsx)(e.code,{children:"--help"}),"\u67e5\u770b\u5e2e\u52a9\u4fe1\u606f:"]}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-bash",children:"$ python main.py --help\n"})}),"\n",(0,a.jsx)(e.h2,{id:"\u547d\u4ee4commands",children:"\u547d\u4ee4(Commands)"}),"\n",(0,a.jsxs)(e.p,{children:["Typer\u5141\u8bb8\u521b\u5efa\u5177\u6709\u591a\u4e2a\u547d\u4ee4(\u4e5f\u79f0\u4e3a\u5b50\u547d\u4ee4)\u7684CLI\u7a0b\u5e8f\u3002\u4f8b\u5982",(0,a.jsx)(e.code,{children:"git"}),"\u5c31\u6709\u591a\u4e2a\u547d\u4ee4\u5982",(0,a.jsx)(e.code,{children:"git push"}),"\u548c",(0,a.jsx)(e.code,{children:"git pull"}),"\u3002"]}),"\n",(0,a.jsx)(e.h3,{id:"\u57fa\u672c\u547d\u4ee4\u793a\u4f8b",children:"\u57fa\u672c\u547d\u4ee4\u793a\u4f8b"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'import typer\r\n\r\napp = typer.Typer()\r\n\r\n@app.command()\r\ndef create():\r\n    print("\u521b\u5efa\u7528\u6237: \u5f20\u4e09")\r\n\r\n@app.command()\r\ndef delete():\r\n    print("\u5220\u9664\u7528\u6237: \u5f20\u4e09")\r\n\r\nif __name__ == "__main__":\r\n    app()\n'})}),"\n",(0,a.jsx)(e.p,{children:"\u8fd0\u884c:"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-bash",children:"# \u67e5\u770b\u5e2e\u52a9\u4fe1\u606f\r\n$ python main.py --help\r\nUsage: main.py [OPTIONS] COMMAND [ARGS]...\r\n\r\nOptions:\r\n  --help  \u663e\u793a\u5e2e\u52a9\u4fe1\u606f\u5e76\u9000\u51fa\r\n\r\nCommands:\r\n  create  \u521b\u5efa\u7528\u6237\r\n  delete  \u5220\u9664\u7528\u6237\r\n\r\n# \u6267\u884ccreate\u547d\u4ee4\r\n$ python main.py create\r\n\u521b\u5efa\u7528\u6237: \u5f20\u4e09\r\n\r\n# \u6267\u884cdelete\u547d\u4ee4\r\n$ python main.py delete\r\n\u5220\u9664\u7528\u6237: \u5f20\u4e09\n"})}),"\n",(0,a.jsx)(e.h3,{id:"\u81ea\u52a8\u663e\u793a\u5e2e\u52a9\u4fe1\u606f",children:"\u81ea\u52a8\u663e\u793a\u5e2e\u52a9\u4fe1\u606f"}),"\n",(0,a.jsxs)(e.p,{children:["\u901a\u8fc7\u8bbe\u7f6e",(0,a.jsx)(e.code,{children:"no_args_is_help=True"}),",\u5f53\u4e0d\u5e26\u53c2\u6570\u8fd0\u884c\u7a0b\u5e8f\u65f6\u4f1a\u81ea\u52a8\u663e\u793a\u5e2e\u52a9\u4fe1\u606f:"]}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'import typer\r\n\r\napp = typer.Typer(no_args_is_help=True)  # \u6dfb\u52a0\u8fd9\u4e2a\u53c2\u6570\r\n\r\n@app.command()\r\ndef create():\r\n    print("\u521b\u5efa\u7528\u6237: \u5f20\u4e09")\r\n\r\n@app.command()\r\ndef delete():\r\n    print("\u5220\u9664\u7528\u6237: \u5f20\u4e09")\r\n\r\nif __name__ == "__main__":\r\n    app()\n'})}),"\n",(0,a.jsx)(e.p,{children:"\u73b0\u5728\u76f4\u63a5\u8fd0\u884c\u7a0b\u5e8f\u5c31\u4f1a\u663e\u793a\u5e2e\u52a9\u4fe1\u606f:"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-bash",children:"$ python main.py\r\nUsage: main.py [OPTIONS] COMMAND [ARGS]...\r\n...\n"})}),"\n",(0,a.jsx)(e.h3,{id:"\u547d\u4ee4\u6392\u5e8f",children:"\u547d\u4ee4\u6392\u5e8f"}),"\n",(0,a.jsx)(e.p,{children:"Typer\u4f1a\u6309\u7167\u547d\u4ee4\u5728\u4ee3\u7801\u4e2d\u58f0\u660e\u7684\u987a\u5e8f\u663e\u793a\u547d\u4ee4\u3002\u4f8b\u5982:"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'import typer\r\n\r\napp = typer.Typer()\r\n\r\n@app.command()\r\ndef delete():  # delete\u5728\u524d\r\n    print("\u5220\u9664\u7528\u6237")\r\n\r\n@app.command() \r\ndef create():  # create\u5728\u540e\r\n    print("\u521b\u5efa\u7528\u6237")\r\n\r\nif __name__ == "__main__":\r\n    app()\n'})}),"\n",(0,a.jsx)(e.p,{children:"\u5e2e\u52a9\u4fe1\u606f\u4e2d\u547d\u4ee4\u7684\u663e\u793a\u987a\u5e8f\u5c06\u662f:"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-bash",children:"Commands:\r\n  delete\r\n  create\n"})})]})}function o(n={}){const{wrapper:e}={...(0,i.R)(),...n.components};return e?(0,a.jsx)(e,{...n,children:(0,a.jsx)(p,{...n})}):p(n)}},5658:(n,e,r)=>{r.d(e,{R:()=>s,x:()=>d});var l=r(758);const a={},i=l.createContext(a);function s(n){const e=l.useContext(i);return l.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function d(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(a):n.components||a:s(n.components),l.createElement(i.Provider,{value:e},n.children)}}}]);