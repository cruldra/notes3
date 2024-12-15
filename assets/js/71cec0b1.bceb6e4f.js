"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[3653],{3434:(n,e,s)=>{s.r(e),s.d(e,{assets:()=>l,contentTitle:()=>u,default:()=>i,frontMatter:()=>c,metadata:()=>r,toc:()=>p});const r=JSON.parse('{"id":"Python/BuiltInModules/subprocess","title":"subprocess","description":"\u793a\u4f8b","source":"@site/docs/Python/BuiltInModules/subprocess.mdx","sourceDirName":"Python/BuiltInModules","slug":"/Python/BuiltInModules/subprocess","permalink":"/notes3/docs/Python/BuiltInModules/subprocess","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/Python/BuiltInModules/subprocess.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"python","previous":{"title":"asyncio","permalink":"/notes3/docs/Python/BuiltInModules/asyncio"},"next":{"title":"\u5e38\u7528\u5e93","permalink":"/notes3/docs/category/\u5e38\u7528\u5e93-4"}}');var o=s(6070),t=s(5658);const c={},u=void 0,l={},p=[{value:"Popen",id:"popen",level:2}];function d(n){const e={code:"code",h2:"h2",p:"p",pre:"pre",strong:"strong",...(0,t.R)(),...n.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'# subprocess\u6a21\u5757\r\n"""\r\n\u5e26\u6709\u53ef\u8bbf\u95eeI/O\u6d41\u7684\u5b50\u8fdb\u7a0b\u6a21\u5757\r\n\u6b64\u6a21\u5757\u5141\u8bb8\u4f60\u521b\u5efa\u8fdb\u7a0b\uff0c\u8fde\u63a5\u5b83\u4eec\u7684\u8f93\u5165/\u8f93\u51fa/\u9519\u8bef\u7ba1\u9053\uff0c\u5e76\u83b7\u53d6\u5b83\u4eec\u7684\u8fd4\u56de\u7801\u3002\r\n"""\r\n\r\n# \u4e3b\u8981 API\r\nrun(...)          # \u8fd0\u884c\u547d\u4ee4\u5e76\u7b49\u5f85\u5b8c\u6210\uff0c\u8fd4\u56deCompletedProcess\u5b9e\u4f8b\r\nPopen(...)        # \u7528\u4e8e\u7075\u6d3b\u5730\u5728\u65b0\u8fdb\u7a0b\u4e2d\u6267\u884c\u547d\u4ee4\u7684\u7c7b\r\n\r\n# \u5e38\u91cf\r\nDEVNULL           # \u7279\u6b8a\u503c\uff0c\u8868\u793a\u4f7f\u7528os.devnull\r\nPIPE              # \u7279\u6b8a\u503c\uff0c\u8868\u793a\u5e94\u8be5\u521b\u5efa\u7ba1\u9053\r\nSTDOUT            # \u7279\u6b8a\u503c\uff0c\u8868\u793astderr\u5e94\u8be5\u91cd\u5b9a\u5411\u5230stdout\r\n\r\n# \u65e7\u7248 API\r\ncall(...)         # \u8fd0\u884c\u547d\u4ee4\u5e76\u7b49\u5f85\u5b8c\u6210\uff0c\u8fd4\u56de\u8fd4\u56de\u7801\r\ncheck_call(...)   # \u4e0ecall()\u76f8\u540c\uff0c\u4f46\u5982\u679c\u8fd4\u56de\u7801\u4e0d\u4e3a0\u5219\u629b\u51faCalledProcessError\r\ncheck_output(...) # \u4e0echeck_call()\u76f8\u540c\uff0c\u4f46\u8fd4\u56destdout\u5185\u5bb9\u800c\u4e0d\u662f\u8fd4\u56de\u7801\r\ngetoutput(...)    # \u5728shell\u4e2d\u8fd0\u884c\u547d\u4ee4\u5e76\u7b49\u5f85\u5b8c\u6210\uff0c\u8fd4\u56de\u8f93\u51fa\r\ngetstatusoutput(...)  # \u5728shell\u4e2d\u8fd0\u884c\u547d\u4ee4\u5e76\u7b49\u5f85\u5b8c\u6210\uff0c\u8fd4\u56de(\u9000\u51fa\u7801,\u8f93\u51fa)\u5143\u7ec4\n'})}),"\n",(0,o.jsx)(e.p,{children:(0,o.jsx)(e.strong,{children:"\u793a\u4f8b"})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:"# \u63a8\u8350\u4f7f\u7528 run()\r\nresult = subprocess.run(['ls', '-l'], capture_output=True, text=True)\r\nprint(result.stdout)\r\n\r\n# \u9700\u8981\u66f4\u591a\u63a7\u5236\u65f6\u4f7f\u7528 Popen\r\nprocess = subprocess.Popen(['ping', 'google.com'], stdout=subprocess.PIPE)\r\nwhile True:\r\n    line = process.stdout.readline()\r\n    if not line:\r\n        break\r\n    print(line.decode())\r\n\r\n# \u7b80\u5355\u547d\u4ee4\u53ef\u4ee5\u4f7f\u7528check_output\r\noutput = subprocess.check_output(['echo', 'hello'])\n"})}),"\n",(0,o.jsx)(e.h2,{id:"popen",children:"Popen"}),"\n",(0,o.jsxs)(e.p,{children:[(0,o.jsx)(e.code,{children:"subprocess.Popen"}),"\u662f",(0,o.jsx)(e.code,{children:"Python"}),"\u4e2d\u7528\u4e8e\u521b\u5efa\u548c\u7ba1\u7406\u5b50\u8fdb\u7a0b\u7684\u6838\u5fc3\u7c7b."]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:"Popen(\r\n    args,                    # \u8981\u6267\u884c\u7684\u547d\u4ee4\uff0c\u53ef\u4ee5\u662f\u5b57\u7b26\u4e32\u6216\u5e8f\u5217\r\n    bufsize=-1,             # \u7f13\u51b2\u533a\u5927\u5c0f\uff0c\u9ed8\u8ba4\u7cfb\u7edf\u7f13\u51b2\r\n    executable=None,        # \u53ef\u6267\u884c\u6587\u4ef6\u8def\u5f84\r\n    stdin=None,             # \u6807\u51c6\u8f93\u5165\uff0c\u53ef\u4ee5\u662fPIPE\r\n    stdout=None,            # \u6807\u51c6\u8f93\u51fa\uff0c\u53ef\u4ee5\u662fPIPE\r\n    stderr=None,            # \u6807\u51c6\u9519\u8bef\uff0c\u53ef\u4ee5\u662fPIPE\r\n    preexec_fn=None,        # \u5b50\u8fdb\u7a0b\u8fd0\u884c\u524d\u7684\u56de\u8c03\u51fd\u6570\r\n    close_fds=True,         # \u662f\u5426\u5173\u95ed\u7236\u8fdb\u7a0b\u7684\u6587\u4ef6\u63cf\u8ff0\u7b26\r\n    shell=False,            # \u662f\u5426\u901a\u8fc7shell\u6267\u884c\r\n    cwd=None,              # \u5b50\u8fdb\u7a0b\u5de5\u4f5c\u76ee\u5f55\r\n    env=None,              # \u5b50\u8fdb\u7a0b\u73af\u5883\u53d8\u91cf\r\n    universal_newlines=None, # \u6587\u672c\u6a21\u5f0f(\u5df2\u5f03\u7528,\u7528text\u4ee3\u66ff)\r\n    startupinfo=None,       # Windows\u4e13\u7528\u542f\u52a8\u4fe1\u606f\r\n    creationflags=0,        # Windows\u4e13\u7528\u521b\u5efa\u6807\u5fd7\r\n    restore_signals=True,    # \u662f\u5426\u6062\u590d\u4fe1\u53f7\u5904\u7406\u5668\r\n    start_new_session=False, # \u662f\u5426\u542f\u52a8\u65b0\u4f1a\u8bdd\r\n    pass_fds=(),            # \u8981\u4f20\u9012\u7ed9\u5b50\u8fdb\u7a0b\u7684\u6587\u4ef6\u63cf\u8ff0\u7b26\r\n\r\n    # \u5173\u952e\u5b57\u53c2\u6570\r\n    text=None,              # \u662f\u5426\u4ee5\u6587\u672c\u6a21\u5f0f\u8fd0\u884c\r\n    encoding='utf-8',       # \u6587\u672c\u7f16\u7801\r\n    errors=None,            # \u7f16\u7801\u9519\u8bef\u5904\u7406\r\n    user=None,              # \u4ee5\u6307\u5b9a\u7528\u6237\u8fd0\u884c\r\n    group=None,             # \u4ee5\u6307\u5b9a\u7ec4\u8fd0\u884c\r\n    extra_groups=None,      # \u989d\u5916\u7684\u7ec4\r\n    umask=-1,              # \u8bbe\u7f6eumask\r\n    pipesize=-1,           # \u7ba1\u9053\u5927\u5c0f\r\n    process_group=None      # \u8fdb\u7a0b\u7ec4\r\n)\n"})}),"\n",(0,o.jsx)(e.p,{children:(0,o.jsx)(e.strong,{children:"\u793a\u4f8b"})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:"# \u57fa\u672c\u4f7f\u7528\r\np = subprocess.Popen(['ls', '-l'], stdout=subprocess.PIPE)\r\noutput = p.communicate()[0]\r\n\r\n# shell\u65b9\u5f0f\r\np = subprocess.Popen('echo $HOME', shell=True, stdout=subprocess.PIPE)\r\n\r\n# \u91cd\u5b9a\u5411\u8f93\u5165\u8f93\u51fa\r\nwith open('output.txt', 'w') as f:\r\n    p = subprocess.Popen(['command'], stdout=f)\r\n\r\n# \u6307\u5b9a\u5de5\u4f5c\u76ee\u5f55\u548c\u73af\u5883\u53d8\u91cf\r\np = subprocess.Popen('command', cwd='/tmp', env={'PATH': '/usr/bin'})\n"})})]})}function i(n={}){const{wrapper:e}={...(0,t.R)(),...n.components};return e?(0,o.jsx)(e,{...n,children:(0,o.jsx)(d,{...n})}):d(n)}},5658:(n,e,s)=>{s.d(e,{R:()=>c,x:()=>u});var r=s(758);const o={},t=r.createContext(o);function c(n){const e=r.useContext(t);return r.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function u(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(o):n.components||o:c(n.components),r.createElement(t.Provider,{value:e},n.children)}}}]);