"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[8125],{2730:(n,e,r)=>{r.r(e),r.d(e,{assets:()=>c,contentTitle:()=>a,default:()=>p,frontMatter:()=>t,metadata:()=>i,toc:()=>l});const i=JSON.parse('{"id":"FrontEnd/Miniapp/Uniapp/\u5fae\u4fe1/\u767b\u5f55","title":"\u767b\u5f55","description":"\u4e3b\u8981\u6d41\u7a0b","source":"@site/docs/FrontEnd/Miniapp/Uniapp/\u5fae\u4fe1/\u767b\u5f55.mdx","sourceDirName":"FrontEnd/Miniapp/Uniapp/\u5fae\u4fe1","slug":"/FrontEnd/Miniapp/Uniapp/\u5fae\u4fe1/\u767b\u5f55","permalink":"/notes3/docs/FrontEnd/Miniapp/Uniapp/\u5fae\u4fe1/\u767b\u5f55","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/FrontEnd/Miniapp/Uniapp/\u5fae\u4fe1/\u767b\u5f55.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"frontEnd","previous":{"title":"\u5fae\u4fe1","permalink":"/notes3/docs/category/\u5fae\u4fe1"},"next":{"title":"\u8e29\u8fc7\u7684\u5751","permalink":"/notes3/docs/FrontEnd/Miniapp/Uniapp/\u8e29\u8fc7\u7684\u5751"}}');var s=r(6070),o=r(5658);const t={},a=void 0,c={},l=[{value:"\u4e3b\u8981\u6d41\u7a0b",id:"\u4e3b\u8981\u6d41\u7a0b",level:2},{value:"\u524d\u7aef",id:"\u524d\u7aef",level:2},{value:"\u540e\u7aef",id:"\u540e\u7aef",level:2}];function d(n){const e={a:"a",code:"code",h2:"h2",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",...(0,o.R)(),...n.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(e.h2,{id:"\u4e3b\u8981\u6d41\u7a0b",children:"\u4e3b\u8981\u6d41\u7a0b"}),"\n",(0,s.jsx)(e.p,{children:"\u5fae\u4fe1\u5c0f\u7a0b\u5e8f\u5b9e\u73b0\u767b\u5f55\u4e3b\u8981\u6d41\u7a0b\u5982\u56fe:"}),"\n",(0,s.jsx)(e.p,{children:(0,s.jsx)(e.img,{src:"https://github.com/cruldra/picx-images-hosting/raw/master/image.8ojq2f5yst.webp",alt:""})}),"\n",(0,s.jsx)(e.h2,{id:"\u524d\u7aef",children:"\u524d\u7aef"}),"\n",(0,s.jsxs)(e.ol,{children:["\n",(0,s.jsx)(e.li,{children:"\u5148\u521b\u5efa\u6309\u94ae"}),"\n"]}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-html",children:'<button\r\n  v-if="isAgree"\r\n  class="wechat-login-btn"\r\n  :loading="loading"\r\n  open-type="getPhoneNumber"\r\n  @getphonenumber="getPhoneNumber"\r\n>\r\n\u5fae\u4fe1\u4e00\u952e\u767b\u5f55\r\n</button>\r\n<button\r\n  v-else\r\n  class="wechat-login-btn"\r\n  @click="handleDisabledClick"\r\n>\r\n\u5fae\u4fe1\u4e00\u952e\u767b\u5f55\r\n</button>\n'})}),"\n",(0,s.jsxs)(e.ol,{start:"2",children:["\n",(0,s.jsxs)(e.li,{children:["\u5728",(0,s.jsx)(e.code,{children:"script"}),"\u4e2d\u5b9a\u4e49",(0,s.jsx)(e.code,{children:"getPhoneNumber"}),"\u65b9\u6cd5"]}),"\n"]}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-js",children:"const getPhoneNumber = async (e: any) => {\r\n  // \u7528\u6237\u62d2\u7edd\u6388\u6743\r\n  if (e.detail.errMsg !== 'getPhoneNumber:ok') {\r\n    await uni.showToast({\r\n      title: '\u767b\u5f55\u5931\u8d25',\r\n      icon: 'none'\r\n    })\r\n    return\r\n  }\r\n\r\n  loading.value = true\r\n  try {\r\n    // 1. \u83b7\u53d6\u767b\u5f55code\r\n    const loginRes = await uni.login()\r\n    // 2. \u8c03\u7528\u540e\u7aef\u63a5\u53e3\uff0c\u4f20\u5165code\u548c\u52a0\u5bc6\u6570\u636e\r\n    const res = (await authService.wxLogin({\r\n      code: loginRes.code,\r\n      encryptedData: e.detail.encryptedData,\r\n      iv: e.detail.iv\r\n    }))\r\n\r\n\r\n    // 3. \u4fdd\u5b58\u767b\u5f55\u6001\r\n    const {token, userInfo} = res.data!!\r\n    uni.setStorageSync('token', token)\r\n    uni.setStorageSync('userInfo', userInfo)\r\n\r\n    // 4. \u767b\u5f55\u6210\u529f\u8df3\u8f6c\r\n    await uni.switchTab({\r\n      url: '/pages/index/index'\r\n    })\r\n\r\n  } catch (error) {\r\n    await uni.showToast({\r\n      title: '\u767b\u5f55\u5931\u8d25\uff0c\u8bf7\u91cd\u8bd5',\r\n      icon: 'none'\r\n    })\r\n  } finally {\r\n    loading.value = false\r\n  }\r\n}\r\nconst handleDisabledClick = () => {\r\n  uni.showToast({\r\n    title: '\u8bf7\u5148\u540c\u610f\u7528\u6237\u534f\u8bae\u548c\u9690\u79c1\u653f\u7b56',\r\n    icon: 'none'\r\n  })\r\n}\n"})}),"\n",(0,s.jsx)(e.h2,{id:"\u540e\u7aef",children:"\u540e\u7aef"}),"\n",(0,s.jsxs)(e.ol,{children:["\n",(0,s.jsxs)(e.li,{children:["\u5f15\u5165",(0,s.jsx)(e.a,{href:"https://github.com/binarywang/WxJava",children:"\u5fae\u4fe1Java\u5f00\u53d1\u5305"})]}),"\n"]}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-xml",children:"<dependency>\r\n  <groupId>com.github.binarywang</groupId>\r\n  <artifactId>weixin-java-miniapp</artifactId>\r\n  <version>4.6.0</version>\r\n</dependency>\n"})}),"\n",(0,s.jsxs)(e.ol,{start:"2",children:["\n",(0,s.jsxs)(e.li,{children:["\u521b\u5efa",(0,s.jsx)(e.code,{children:"WxMaService"}),"\u5b9e\u4f8b"]}),"\n"]}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-kotlin",children:'val wxMaService: WxMaService = WxMaServiceImpl()\r\nval config = WxMaDefaultConfigImpl()\r\nconfig.appid = "wx123456789"\r\nconfig.secret = "1234567890abcdef"\r\nwxMaService.wxMaConfig = config\n'})}),"\n",(0,s.jsxs)(e.ol,{start:"3",children:["\n",(0,s.jsx)(e.li,{children:"\u83b7\u53d6\u624b\u673a\u53f7\u5b9e\u73b0\u81ea\u52a8\u6ce8\u518c\u767b\u5f55"}),"\n"]}),"\n",(0,s.jsx)(e.pre,{children:(0,s.jsx)(e.code,{className:"language-kotlin",children:'@PostMapping("/auth/wxLogin")\r\nfun wxLogin(ctx: Context) {\r\n    val request = ctx.bodyAsClass<WechatLoginData>()\r\n    log.info("\u4ee3\u7801: ${request.code}")\r\n    // Initialize WxMaService\r\n    val wxService = IOC.getComponent(WxMaService::class.java)\r\n    log.info("appid: ${wxService.wxMaConfig.appid}")\r\n    // Get session info\r\n    val sessionInfo: WxMaJscode2SessionResult = wxService.jsCode2SessionInfo(request.code)\r\n\r\n    // Decrypt user info using sessionKey if needed\r\n    //val userInfo = wxService.userService.getUserInfo(sessionInfo.sessionKey, request.encryptedData, request.iv)\r\n    val wxPhoneInfo = wxService.userService.getPhoneNoInfo(sessionInfo.sessionKey, request.encryptedData, request.iv)\r\n    log.info("openid: ${sessionInfo.openid}")\r\n    ctx.json(\r\n        ResponsePayloads(\r\n            data = WechatLoginResult(\r\n                token = "TODO",\r\n                userInfo = UserInfo(\r\n                    avatar = "userInfo.avatarUrl",\r\n                    nickname = "userInfo.nickName",\r\n                    phoneNumber = wxPhoneInfo.phoneNumber\r\n                )\r\n            )\r\n        )\r\n    )\r\n}\n'})})]})}function p(n={}){const{wrapper:e}={...(0,o.R)(),...n.components};return e?(0,s.jsx)(e,{...n,children:(0,s.jsx)(d,{...n})}):d(n)}},5658:(n,e,r)=>{r.d(e,{R:()=>t,x:()=>a});var i=r(758);const s={},o=i.createContext(s);function t(n){const e=i.useContext(o);return i.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function a(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(s):n.components||s:t(n.components),i.createElement(o.Provider,{value:e},n.children)}}}]);