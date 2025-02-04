"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[4425],{7248:(n,e,r)=>{r.r(e),r.d(e,{assets:()=>i,contentTitle:()=>c,default:()=>p,frontMatter:()=>o,metadata:()=>t,toc:()=>l});const t=JSON.parse('{"id":"AI/\u4e00\u4e9b\u6982\u5ff54","title":"\u4e00\u4e9b\u6982\u5ff54","description":"","source":"@site/docs/AI/\u4e00\u4e9b\u6982\u5ff54.mdx","sourceDirName":"AI","slug":"/AI/\u4e00\u4e9b\u6982\u5ff54","permalink":"/notes3/docs/AI/\u4e00\u4e9b\u6982\u5ff54","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/AI/\u4e00\u4e9b\u6982\u5ff54.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"ai","previous":{"title":"\u4e00\u4e9b\u6982\u5ff53","permalink":"/notes3/docs/AI/\u4e00\u4e9b\u6982\u5ff53"},"next":{"title":"\u5165\u95e8","permalink":"/notes3/docs/AI/\u6df1\u5ea6\u5b66\u4e60/\u5165\u95e8"}}');var s=r(6070),a=r(5658);const o={},c=void 0,i={},l=[];function d(n){const e={code:"code",img:"img",p:"p",pre:"pre",...(0,a.R)(),...n.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-plantuml",children:'@startuml\r\nskinparam backgroundColor #FFFBD0\r\nskinparam componentStyle uml2\r\n\r\npackage "AI\u5f00\u53d1\u6d41\u7a0b" {\r\n  [\u539f\u59cb\u6570\u636e] as raw_data #Pink\r\n  [Dataset\\n(\u6e05\u6d17/\u9884\u5904\u7406)] as dataset #LightBlue\r\n  [Transformers\u6a21\u578b\\n(GPT/BERT\u7b49)] as model #tan\r\n  \r\n  package "\u6a21\u578b\u5fae\u8c03\u65b9\u6cd5" #LightCyan {\r\n    [SFT\u5fae\u8c03\\n(\u4efb\u52a1\u9002\u914d)] as sft #LightGreen\r\n    [LoRA\u5fae\u8c03\\n(\u4f4e\u79e9\u9002\u914d)] as lora #LightPink\r\n  }\r\n  \r\n  package "\u6a21\u578b\u5b58\u50a8\u683c\u5f0f" #AliceBlue {\r\n    [SafeTensors\\n(HuggingFace)] as safetensors #lavender\r\n    [.pth\u6587\u4ef6\\n(PyTorch)] as pth #LightGreen\r\n    [.ckpt\u6587\u4ef6\\n(Lightning AI)] as ckpt #LightYellow\r\n  }\r\n  \r\n  [CUDA\u52a0\u901f\u5e93] as cuda #LightGray\r\n  [GPU\u96c6\u7fa4] as gpu #khaki\r\n\r\n  raw_data --\x3e dataset : \u8f93\u5165\\n(\u6587\u672c/\u56fe\u7247\u7b49)\r\n  dataset --\x3e model : \u5582\u6570\u636e\\n(\u6279\u91cf\u52a0\u8f7d)\r\n  model --\x3e sft : \u5b8c\u6574\u53c2\u6570\\n\u5fae\u8c03\r\n  model --\x3e lora : \u90e8\u5206\u53c2\u6570\\n\u9ad8\u6548\u5fae\u8c03\r\n  sft --\x3e safetensors : \u5b8c\u6574\u6a21\u578b\u4fdd\u5b58\r\n  sft --\x3e pth : \u5feb\u901f\u4fdd\u5b58\\n(\u5f00\u53d1\u8c03\u8bd5\u7528)\r\n  sft --\x3e ckpt : \u4e2d\u95f4\u4fdd\u5b58\\n(\u542b\u4f18\u5316\u5668\u72b6\u6001)\r\n  lora --\x3e safetensors : \u4ec5\u4fdd\u5b58\\n\u9002\u914d\u5668\u6743\u91cd\r\n  safetensors --\x3e model : \u5b89\u5168\u52a0\u8f7d\\n(\u652f\u6301\u8de8\u6846\u67b6)\r\n  pth --\x3e model : \u5feb\u901f\u52a0\u8f7d\\n(\u9700\u4fe1\u4efb\u6765\u6e90)\r\n  ckpt --\x3e model : \u65ad\u70b9\u7eed\u8bad\\n(\u6062\u590d\u8bad\u7ec3\u72b6\u6001)\r\n  cuda -up-> gpu : \u786c\u4ef6\u9a71\u52a8\\n(\u5e76\u884c\u8ba1\u7b97)\r\n  model .down.> cuda : \u8c03\u7528\u77e9\u9635\u8fd0\u7b97\\n(\u81ea\u52a8\u52a0\u901f)\r\n  sft --\x3e gpu : \u53cd\u5411\u4f20\u64ad\\n(\u5168\u91cf\u66f4\u65b0)\r\n  lora --\x3e gpu : \u53cd\u5411\u4f20\u64ad\\n(\u589e\u91cf\u66f4\u65b0)\r\n}\r\n\r\nnote right of model\r\n  **\u5b58\u50a8\u683c\u5f0f\u5bf9\u6bd4**\uff1a\r\n  \u25b8 SafeTensors\uff1a\u9632\u75c5\u6bd2\u8f6f\u4ef6\u6700\u7231\\n    (\u65e0\u4ee3\u7801\u6267\u884c\u98ce\u9669)\r\n  \u25b8 .pth\uff1a\u65b9\u4fbf\u4f46\u5371\u9669\\n    (\u53ef\u80fd\u643a\u5e26\u6076\u610f\u4ee3\u7801)\r\n  \u25b8 .ckpt\uff1a\u4f53\u79ef\u5e9e\u5927\\n    (\u5305\u542b\u8bad\u7ec3\u73b0\u573a\u5feb\u7167)\r\nend note\r\n\r\nnote left of lora\r\n  **\u5fae\u8c03\u65b9\u5f0f\u5bf9\u6bd4**\uff1a\r\n  \u25b8 SFT\uff1a\u5168\u91cf\u53c2\u6570\u66f4\u65b0\\n    (\u8d44\u6e90\u6d88\u8017\u5927)\r\n  \u25b8 LoRA\uff1a\u4f4e\u79e9\u77e9\u9635\u66f4\u65b0\\n    (\u9ad8\u6548\u7701\u663e\u5b58)\r\nend note\r\n\r\nnote left of gpu\r\n  **GPU\u663e\u5b58\u5c0f\u5267\u573a**\uff1a\r\n  \u52a0\u8f7d.pth \u2192 \u4fdd\u5b89\u8981\u68c0\u67e5\u6bcf\u4e2a\u4eba\\n  \uff08\u901f\u5ea6\u6162\uff09\r\n  \u52a0\u8f7d.safetensors \u2192 VIP\u901a\u9053\\n  \u76f4\u63a5\u8fdb\u4f1a\u573a\uff08\u60f0\u6027\u52a0\u8f7d\uff09\r\nend note\r\n@enduml\n'})}),"\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.img,{src:"https://github.com/cruldra/picx-images-hosting/raw/master/image.7zqjm1f0p1.png",alt:""})})]})}function p(n={}){const{wrapper:e}={...(0,a.R)(),...n.components};return e?(0,s.jsx)(e,{...n,children:(0,s.jsx)(d,{...n})}):d(n)}},5658:(n,e,r)=>{r.d(e,{R:()=>o,x:()=>c});var t=r(758);const s={},a=t.createContext(s);function o(n){const e=t.useContext(a);return t.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function c(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(s):n.components||s:o(n.components),t.createElement(a.Provider,{value:e},n.children)}}}]);