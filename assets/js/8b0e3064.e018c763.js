"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[1427],{1640:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>o,contentTitle:()=>c,default:()=>a,frontMatter:()=>d,metadata:()=>t,toc:()=>l});const t=JSON.parse('{"id":"AI/\u6df1\u5ea6\u5b66\u4e60/\u5165\u95e8","title":"\u5165\u95e8","description":"\u8981\u8fd0\u884c\u4e0a\u9762\u7684\u4f8b\u5b50:","source":"@site/docs/AI/\u6df1\u5ea6\u5b66\u4e60/\u5165\u95e8.mdx","sourceDirName":"AI/\u6df1\u5ea6\u5b66\u4e60","slug":"/AI/\u6df1\u5ea6\u5b66\u4e60/\u5165\u95e8","permalink":"/notes3/docs/AI/\u6df1\u5ea6\u5b66\u4e60/\u5165\u95e8","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/AI/\u6df1\u5ea6\u5b66\u4e60/\u5165\u95e8.mdx","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"ai","previous":{"title":"\u4e00\u4e9b\u6982\u5ff5","permalink":"/notes3/docs/AI/\u4e00\u4e9b\u6982\u5ff5"},"next":{"title":"Tokenization\u548cEmbedding","permalink":"/notes3/docs/AI/\u6df1\u5ea6\u5b66\u4e60/Tokenization\u548cEmbedding"}}');var s=r(6070),i=r(5658);const d={sidebar_position:1},c=void 0,o={},l=[];function h(e){const n={a:"a",code:"code",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",...(0,i.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'if __name__ == \'__main__\':\r\n    from transformers import AutoTokenizer, AutoModelForCausalLM\r\n\r\n    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")\r\n    model = AutoModelForCausalLM.from_pretrained("distilgpt2", output_hidden_states=True)\r\n    print(tokenizer)\r\n    print(model) \n'})}),"\n",(0,s.jsx)(n.p,{children:"\u8981\u8fd0\u884c\u4e0a\u9762\u7684\u4f8b\u5b50:"}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["\u4f7f\u7528",(0,s.jsx)(n.code,{children:"nvidia-smi"}),"\u67e5\u770b",(0,s.jsx)(n.a,{href:"/notes3/docs/AI/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/CUDA",children:"CUDA"}),"\u7248\u672c"]}),"\n",(0,s.jsxs)(n.li,{children:["\u5b89\u88c5",(0,s.jsx)(n.code,{children:"pytorch"})," - ",(0,s.jsx)(n.code,{children:"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"})]}),"\n",(0,s.jsxs)(n.li,{children:["\u5b89\u88c5",(0,s.jsx)(n.code,{children:"transformers"})," - ",(0,s.jsx)(n.code,{children:"pip install transformers"})]}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"\u76f8\u5173\u5e93\u5bf9\u6bd4\u8868"})}),"\n",(0,s.jsxs)(n.table,{children:[(0,s.jsx)(n.thead,{children:(0,s.jsxs)(n.tr,{children:[(0,s.jsx)(n.th,{children:"\u5e93\u540d\u79f0"}),(0,s.jsx)(n.th,{children:"\u5b9a\u4e49"}),(0,s.jsx)(n.th,{children:"\u4e3b\u8981\u529f\u80fd/\u7279\u70b9"}),(0,s.jsx)(n.th,{children:"\u5178\u578b\u7528\u9014"}),(0,s.jsx)(n.th,{children:"\u5b89\u88c5\u547d\u4ee4 (PyPI)"}),(0,s.jsx)(n.th,{children:"\u7248\u672c\u517c\u5bb9\u6027 (\u793a\u4f8b)"})]})}),(0,s.jsxs)(n.tbody,{children:[(0,s.jsxs)(n.tr,{children:[(0,s.jsx)(n.td,{children:(0,s.jsx)(n.strong,{children:"CUDA"})}),(0,s.jsx)(n.td,{children:"NVIDIA\u7684\u5e76\u884c\u8ba1\u7b97\u5e73\u53f0\u548c\u7f16\u7a0b\u6a21\u578b"}),(0,s.jsx)(n.td,{children:"GPU\u52a0\u901f\u8ba1\u7b97\uff0c\u63d0\u4f9b\u6df1\u5ea6\u5b66\u4e60\u5e95\u5c42\u786c\u4ef6\u52a0\u901f\u652f\u6301"}),(0,s.jsx)(n.td,{children:"\u52a0\u901f\u795e\u7ecf\u7f51\u7edc\u8bad\u7ec3/\u63a8\u7406\uff0c\u9ad8\u6027\u80fd\u5e76\u884c\u8ba1\u7b97"}),(0,s.jsx)(n.td,{children:"\u9700\u901a\u8fc7NVIDIA\u5b98\u7f51\u5b89\u88c5\u5bf9\u5e94\u7248\u672c"}),(0,s.jsx)(n.td,{children:"\u4e0eGPU\u9a71\u52a8\u7248\u672c\u5f3a\u76f8\u5173"})]}),(0,s.jsxs)(n.tr,{children:[(0,s.jsx)(n.td,{children:(0,s.jsx)(n.strong,{children:"PyTorch"})}),(0,s.jsx)(n.td,{children:"\u57fa\u4e8ePython\u7684\u5f00\u6e90\u673a\u5668\u5b66\u4e60\u6846\u67b6"}),(0,s.jsx)(n.td,{children:"\u52a8\u6001\u8ba1\u7b97\u56fe\uff0c\u81ea\u52a8\u5fae\u5206\uff0cGPU\u52a0\u901f\uff0c\u4e30\u5bcc\u7684\u795e\u7ecf\u7f51\u7edc\u6a21\u5757"}),(0,s.jsx)(n.td,{children:"\u6a21\u578b\u5f00\u53d1/\u8bad\u7ec3\uff0c\u5b66\u672f\u7814\u7a76\uff0c\u751f\u4ea7\u90e8\u7f72"}),(0,s.jsxs)(n.td,{children:[(0,s.jsx)(n.code,{children:"pip install torch torchvision torchaudio"})," (\u9700\u6307\u5b9aCUDA\u7248\u672c)"]}),(0,s.jsx)(n.td,{children:"2.0.1+cu118 (\u5bf9\u5e94CUDA 11.8)"})]}),(0,s.jsxs)(n.tr,{children:[(0,s.jsx)(n.td,{children:(0,s.jsx)(n.strong,{children:"TorchVision"})}),(0,s.jsx)(n.td,{children:"PyTorch\u7684\u8ba1\u7b97\u673a\u89c6\u89c9\u6269\u5c55\u5e93"}),(0,s.jsx)(n.td,{children:"\u63d0\u4f9b\u56fe\u50cf\u6570\u636e\u96c6\u3001\u53d8\u6362\u64cd\u4f5c\u3001\u9884\u8bad\u7ec3\u6a21\u578b\uff08ResNet\u7b49\uff09"}),(0,s.jsx)(n.td,{children:"\u56fe\u50cf\u5206\u7c7b/\u68c0\u6d4b/\u5206\u5272\uff0c\u6570\u636e\u589e\u5f3a\uff0c\u8fc1\u79fb\u5b66\u4e60"}),(0,s.jsx)(n.td,{children:(0,s.jsx)(n.code,{children:"pip install torchvision"})}),(0,s.jsx)(n.td,{children:"\u9700\u4e0ePyTorch\u7248\u672c\u5339\u914d (e.g. 0.15.1+cu118)"})]}),(0,s.jsxs)(n.tr,{children:[(0,s.jsx)(n.td,{children:(0,s.jsx)(n.strong,{children:"TorchAudio"})}),(0,s.jsx)(n.td,{children:"PyTorch\u7684\u97f3\u9891\u5904\u7406\u5e93"}),(0,s.jsx)(n.td,{children:"\u63d0\u4f9b\u97f3\u9891I/O\u3001\u4fe1\u53f7\u5904\u7406\u3001\u9884\u8bad\u7ec3\u8bed\u97f3\u6a21\u578b"}),(0,s.jsx)(n.td,{children:"\u8bed\u97f3\u8bc6\u522b/\u5408\u6210\uff0c\u97f3\u9891\u5206\u7c7b\uff0c\u58f0\u7eb9\u8bc6\u522b"}),(0,s.jsx)(n.td,{children:(0,s.jsx)(n.code,{children:"pip install torchaudio"})}),(0,s.jsx)(n.td,{children:"\u9700\u4e0ePyTorch\u7248\u672c\u5339\u914d (e.g. 2.0.2+cu118)"})]}),(0,s.jsxs)(n.tr,{children:[(0,s.jsx)(n.td,{children:(0,s.jsx)(n.strong,{children:"Transformers"})}),(0,s.jsx)(n.td,{children:"Hugging Face\u7684\u81ea\u7136\u8bed\u8a00\u5904\u7406\u5e93"}),(0,s.jsx)(n.td,{children:"\u63d0\u4f9bBERT/GPT\u7b49\u9884\u8bad\u7ec3\u6a21\u578b\uff0c\u652f\u6301\u6587\u672c\u5206\u7c7b/\u751f\u6210/\u7ffb\u8bd1\u7b49NLP\u4efb\u52a1"}),(0,s.jsx)(n.td,{children:"\u6587\u672c\u751f\u6210\uff0c\u95ee\u7b54\u7cfb\u7edf\uff0c\u60c5\u611f\u5206\u6790\uff0c\u673a\u5668\u7ffb\u8bd1"}),(0,s.jsx)(n.td,{children:(0,s.jsx)(n.code,{children:"pip install transformers"})}),(0,s.jsx)(n.td,{children:"4.30.0+ (\u901a\u5e38\u5411\u524d\u517c\u5bb9\u4e3b\u6d41PyTorch\u7248\u672c)"})]})]})]}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"\u67b6\u6784\u56fe"})}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{src:"https://github.com/cruldra/picx-images-hosting/raw/master/image.1ovji4pgaa.png",alt:""})})]})}function a(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(h,{...e})}):h(e)}},5658:(e,n,r)=>{r.d(n,{R:()=>d,x:()=>c});var t=r(758);const s={},i=t.createContext(s);function d(e){const n=t.useContext(i);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:d(e.components),t.createElement(i.Provider,{value:n},e.children)}}}]);