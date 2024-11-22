"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[6555],{5184:(e,r,n)=>{n.r(r),n.d(r,{assets:()=>c,contentTitle:()=>l,default:()=>p,frontMatter:()=>s,metadata:()=>t,toc:()=>d});const t=JSON.parse('{"id":"JVM/Libraries/Playwright","title":"Playwright","description":"Playwright\u662f\u4e00\u4e2a\u7528\u4e8e\u81ea\u52a8\u5316\u6d4f\u89c8\u5668\u7684\u5e93,\u7c7b\u4f3cSelenium,\u652f\u6301Chrome\u3001Firefox\u3001Safari\u548c","source":"@site/docs/JVM/Libraries/Playwright.mdx","sourceDirName":"JVM/Libraries","slug":"/JVM/Libraries/Playwright","permalink":"/notes3/docs/JVM/Libraries/Playwright","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/JVM/Libraries/Playwright.mdx","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"jvmEcosystem","previous":{"title":"EBean","permalink":"/notes3/docs/JVM/Libraries/ORM/EBean"},"next":{"title":"\u65e5\u5fd7","permalink":"/notes3/docs/category/\u65e5\u5fd7"}}');var i=n(6070),a=n(5658),o=n(7758);const s={sidebar_position:1},l=void 0,c={},d=[{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2},{value:"\u901a\u8fc7<code>CDP</code>\u8fde\u63a5\u6d4f\u89c8\u5668",id:"\u901a\u8fc7cdp\u8fde\u63a5\u6d4f\u89c8\u5668",level:2},{value:"\u53cd\u673a\u5668\u4eba\u68c0\u6d4b",id:"\u53cd\u673a\u5668\u4eba\u68c0\u6d4b",level:2},{value:"\u53c2\u8003",id:"\u53c2\u8003",level:3},{value:"\u67e5\u627e\u9875\u9762\u4e0a\u7684\u5143\u7d20",id:"\u67e5\u627e\u9875\u9762\u4e0a\u7684\u5143\u7d20",level:2},{value:"\u67e5\u627e\u591a\u4e2a\u5143\u7d20",id:"\u67e5\u627e\u591a\u4e2a\u5143\u7d20",level:3},{value:"\u67e5\u627e\u5355\u4e2a\u5143\u7d20",id:"\u67e5\u627e\u5355\u4e2a\u5143\u7d20",level:3},{value:"\u67e5\u627e\u5305\u542b\u7279\u5b9a\u6587\u672c\u7684\u5143\u7d20",id:"\u67e5\u627e\u5305\u542b\u7279\u5b9a\u6587\u672c\u7684\u5143\u7d20",level:3}];function h(e){const r={a:"a",code:"code",h2:"h2",h3:"h3",img:"img",li:"li",p:"p",pre:"pre",ul:"ul",...(0,a.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsxs)(r.p,{children:[(0,i.jsx)(r.a,{href:"https://playwright.dev/",children:"Playwright"}),"\u662f\u4e00\u4e2a\u7528\u4e8e\u81ea\u52a8\u5316\u6d4f\u89c8\u5668\u7684\u5e93,\u7c7b\u4f3c",(0,i.jsx)(r.code,{children:"Selenium"}),",\u652f\u6301",(0,i.jsx)(r.code,{children:"Chrome"}),"\u3001",(0,i.jsx)(r.code,{children:"Firefox"}),"\u3001",(0,i.jsx)(r.code,{children:"Safari"}),"\u548c\r\n",(0,i.jsx)(r.code,{children:"Microsoft Edge"}),"\u7b49\u4e3b\u6d41\u6d4f\u89c8\u5668."]}),"\n",(0,i.jsx)(r.h2,{id:"\u5b89\u88c5",children:"\u5b89\u88c5"}),"\n",(0,i.jsx)(r.pre,{children:(0,i.jsx)(r.code,{className:"language-xml",children:"<dependency>\r\n    <groupId>com.microsoft.playwright</groupId>\r\n    <artifactId>playwright</artifactId>\r\n    <version>1.47.0</version>\r\n</dependency>\n"})}),"\n",(0,i.jsxs)(r.h2,{id:"\u901a\u8fc7cdp\u8fde\u63a5\u6d4f\u89c8\u5668",children:["\u901a\u8fc7",(0,i.jsx)(r.code,{children:"CDP"}),"\u8fde\u63a5\u6d4f\u89c8\u5668"]}),"\n",(0,i.jsxs)(r.p,{children:[(0,i.jsx)(r.a,{href:"https://chromedevtools.github.io/devtools-protocol/",children:"Chrome DevTools Protocol"}),"\u662f",(0,i.jsx)(r.code,{children:"Chrome"}),"\u6d4f\u89c8\u5668\u63d0\u4f9b\u7684\u4e00\u5957\u8c03\u8bd5\u534f\u8bae."]}),"\n",(0,i.jsx)(r.pre,{children:(0,i.jsx)(r.code,{className:"language-kotlin",children:'@Test\r\nfun test1() {\r\n    val cdpPort = 9222\r\n    val executable = "C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe"\r\n    //1.\u51c6\u5907\u7528\u6237\u6570\u636e\u76ee\u5f55\r\n    val customerDataDir = "D:\\\\ProgramData\\\\Chrome129".toFile()\r\n    //2.\u68c0\u67e5cdp\u7aef\u53e3\u662f\u5426\u53ef\u7528\r\n    isPortAvailable(cdpPort).raiseIfNotTrue(\r\n        "[${cdpPort}]\u7aef\u53e3\u88ab\u5360\u7528",\r\n        Codes.PORT_IS_OCCUPIED_CODE\r\n    )\r\n    //3.\u542f\u52a8chrome\r\n    async {\r\n        stringBuilder(executable) {\r\n            //appendWithSpace("--window-size=${chromeProperties.windowWidth},${chromeProperties.windowHeight}")\r\n            //appendWithSpace("""--user-agent="${chromeProperties.userAgent}" """)\r\n            appendWithSpace("--remote-debugging-port=${cdpPort}")\r\n            appendWithSpace("--user-data-dir=${customerDataDir.absolutePath}")\r\n            appendWithSpace("--no-default-browser-check")\r\n        }.toString().asCommandLine().exec()\r\n    }\r\n    retry(30) {\r\n        isPortAvailable(cdpPort).raiseIfTrue("CDP\u7aef\u53e3\u672a\u6253\u5f00", 129062)\r\n    }\r\n    val cdpUrl = "http://localhost:${cdpPort}"\r\n    //4.\u4f7f\u7528playwright\u5b8c\u6210\u91c7\u96c6\r\n    return playwright {\r\n        chromeOverCDP(cdpUrl) {\r\n            page(\r\n                "https://www.browserscan.net/zh",\r\n                listOf(AntiCrawlerDetectionPlugin())\r\n            ) {\r\n\r\n                sleep(1000)\r\n            }\r\n        }\r\n    }\r\n}\n'})}),"\n",(0,i.jsx)(r.h2,{id:"\u53cd\u673a\u5668\u4eba\u68c0\u6d4b",children:"\u53cd\u673a\u5668\u4eba\u68c0\u6d4b"}),"\n",(0,i.jsxs)(r.p,{children:["\u901a\u8fc7",(0,i.jsx)(r.code,{children:"CDP"}),"\u8fde\u63a5\u5230\u6d4f\u89c8\u5668\u6253\u5f00",(0,i.jsx)(r.a,{href:"https://www.browserscan.net/zh",children:"\u6307\u7eb9\u6d4b\u8bd5\u9875\u9762"}),"\u53ef\u4ee5\u770b\u5230\u9875\u9762\u88ab\u68c0\u6d4b\u4e3a",(0,i.jsx)(r.code,{children:"\u673a\u5668\u4eba"}),"\u8bbf\u95ee"]}),"\n",(0,i.jsx)(r.p,{children:(0,i.jsx)(r.img,{src:"https://github.com/cruldra/picx-images-hosting/raw/master/image.45ofnsun6.webp",alt:""})}),"\n",(0,i.jsx)(r.p,{children:"\u9700\u8981\u5728\u9875\u9762\u52a0\u8f7d\u4e4b\u524d\u8fd0\u884c\u4ee5\u4e0b\u811a\u672c:"}),"\n",(0,i.jsx)(r.pre,{children:(0,i.jsx)(r.code,{className:"language-kotlin",children:'fun <R> Browser.page(\r\n    url: String,\r\n    plugins: List<Plugin> = emptyList(),\r\n    block: Page.() -> R\r\n): R {\r\n    val page = contexts().first().newPage()\r\n    val preventCdpCheckScript = """ (() => {\r\n            const originalError = Error;\r\n            function CustomError(...args) {\r\n                const error = new originalError(...args);\r\n                Object.defineProperty(error, \'stack\', {\r\n                    get: () => \'\',\r\n                    configurable: false\r\n                });\r\n                return error;\r\n            }\r\n            CustomError.prototype = originalError.prototype;\r\n            window.Error = CustomError;\r\n\r\n            const observer = new MutationObserver(() => {\r\n                if (window.Error !== CustomError) {\r\n                    window.Error = CustomError;\r\n                }\r\n            });\r\n            observer.observe(document, { childList: true, subtree: true });\r\n        })();"""\r\n    return page.use {\r\n        plugins.onEach { plugin ->\r\n            plugin.apply(page)\r\n        }\r\n        page.evaluate(preventCdpCheckScript)\r\n        page.onLoad {\r\n            it.evaluate(preventCdpCheckScript)\r\n        }\r\n        page.navigate(url)\r\n        block(page)\r\n    }\r\n}\n'})}),"\n",(0,i.jsx)(r.h3,{id:"\u53c2\u8003",children:"\u53c2\u8003"}),"\n",(0,i.jsxs)(r.ul,{children:["\n",(0,i.jsx)(r.li,{children:(0,i.jsx)(r.a,{href:"https://github.com/DedInc/pystealth",children:"DedInc/pystealth: Python module for preventing detection of CDP in Selenium, Puppeteer, and Playwright."})}),"\n",(0,i.jsx)(r.li,{children:(0,i.jsx)(r.a,{href:"https://github.com/ultrafunkamsterdam/nodriver",children:"ultrafunkamsterdam/nodriver: Successor of Undetected-Chromedriver. Providing a blazing fast framework for web automation, webscraping, bots and any other creative ideas which are normally hindered by annoying anti bot systems like Captcha / CloudFlare / Imperva / hCaptcha"})}),"\n"]}),"\n",(0,i.jsx)(r.h2,{id:"\u67e5\u627e\u9875\u9762\u4e0a\u7684\u5143\u7d20",children:"\u67e5\u627e\u9875\u9762\u4e0a\u7684\u5143\u7d20"}),"\n",(0,i.jsx)(r.h3,{id:"\u67e5\u627e\u591a\u4e2a\u5143\u7d20",children:"\u67e5\u627e\u591a\u4e2a\u5143\u7d20"}),"\n",(0,i.jsx)(r.pre,{children:(0,i.jsx)(r.code,{className:"language-kotlin",children:'val elements = page.querySelectorAll("div.card-body")\n'})}),"\n",(0,i.jsx)(r.h3,{id:"\u67e5\u627e\u5355\u4e2a\u5143\u7d20",children:"\u67e5\u627e\u5355\u4e2a\u5143\u7d20"}),"\n",(0,i.jsx)(r.pre,{children:(0,i.jsx)(r.code,{className:"language-kotlin",children:'val element = page.querySelector("div.card-body")\n'})}),"\n",(0,i.jsx)(r.h3,{id:"\u67e5\u627e\u5305\u542b\u7279\u5b9a\u6587\u672c\u7684\u5143\u7d20",children:"\u67e5\u627e\u5305\u542b\u7279\u5b9a\u6587\u672c\u7684\u5143\u7d20"}),"\n",(0,i.jsx)(r.pre,{children:(0,i.jsx)(r.code,{className:"language-kotlin",children:'val locator = getByText("\u5df2\u7ecf\u5230\u5e95\u4e86\uff0c\u6ca1\u6709\u66f4\u591a\u5185\u5bb9\u4e86")\n'})}),"\n","\n",(0,i.jsx)(o.R2,{children:(0,i.jsxs)(r.p,{children:[(0,i.jsx)(r.code,{children:"getByText"}),"\u8fd4\u56de\u7684\u662f\u4e00\u4e2a",(0,i.jsx)(r.a,{href:"https://playwright.dev/docs/api/class-locator",children:"Locator"}),"\u5bf9\u8c61,\u5173\u4e8e\u5b83\u548c",(0,i.jsx)(r.a,{href:"https://playwright.dev/docs/api/class-elementhandle",children:"ElementHandle"}),"\u7684\u533a\u522b\u53ef\u4ee5\u53c2\u8003",(0,i.jsx)(r.a,{href:"https://poe.com/s/55GY7CXenvqwevo7RscG",children:"\u8fd9\u91cc"})]})})]})}function p(e={}){const{wrapper:r}={...(0,a.R)(),...e.components};return r?(0,i.jsx)(r,{...e,children:(0,i.jsx)(h,{...e})}):h(e)}},7758:(e,r,n)=>{n.d(r,{R2:()=>o});n(758);var t=n(4731),i=n(4675),a=n(6070);function o(e){let{children:r,title:n="\u63d0\u793a"}=e;return(0,a.jsx)(t.F,{variant:"light",color:"blue",title:n,icon:(0,a.jsx)(i.Q9E,{size:18}),children:r})}}}]);