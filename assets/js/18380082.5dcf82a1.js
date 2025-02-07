"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[1287],{3361:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>c,contentTitle:()=>o,default:()=>d,frontMatter:()=>i,metadata:()=>r,toc:()=>l});const r=JSON.parse('{"id":"Python/Libraries/\u963f\u91cc\u4e91/\u77ed\u4fe1","title":"\u77ed\u4fe1","description":"\u963f\u91cc\u4e91\u77ed\u4fe1\u670d\u52a1SDK","source":"@site/docs/Python/Libraries/\u963f\u91cc\u4e91/\u77ed\u4fe1.mdx","sourceDirName":"Python/Libraries/\u963f\u91cc\u4e91","slug":"/Python/Libraries/\u963f\u91cc\u4e91/\u77ed\u4fe1","permalink":"/notes3/docs/Python/Libraries/\u963f\u91cc\u4e91/\u77ed\u4fe1","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/Python/Libraries/\u963f\u91cc\u4e91/\u77ed\u4fe1.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"python","previous":{"title":"OSS","permalink":"/notes3/docs/Python/Libraries/\u963f\u91cc\u4e91/OSS"},"next":{"title":"ASGI\u670d\u52a1\u5668","permalink":"/notes3/docs/category/asgi\u670d\u52a1\u5668"}}');var t=s(6070),a=s(5658);const i={},o=void 0,c={},l=[{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2},{value:"\u793a\u4f8b",id:"\u793a\u4f8b",level:2}];function p(e){const n={a:"a",code:"code",h2:"h2",p:"p",pre:"pre",...(0,a.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.a,{href:"https://help.aliyun.com/zh/sms/developer-reference/sdk-product-overview/",children:"\u963f\u91cc\u4e91\u77ed\u4fe1\u670d\u52a1SDK"})}),"\n",(0,t.jsx)(n.h2,{id:"\u5b89\u88c5",children:"\u5b89\u88c5"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:"pip install alibabacloud_dysmsapi20170525\n"})}),"\n",(0,t.jsx)(n.h2,{id:"\u793a\u4f8b",children:"\u793a\u4f8b"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client\r\nfrom alibabacloud_dysmsapi20170525.models import SendSmsRequest\r\nfrom alibabacloud_tea_openapi import models as open_api_models\r\n\r\nclass SMSClient:\r\n    def __init__(self):\r\n        config = open_api_models.Config(\r\n            access_key_id=settings.aliyun_access_key_id,\r\n            access_key_secret=settings.aliyun_access_key_secret,\r\n        )\r\n        # \u8bbf\u95ee\u7684\u57df\u540d\r\n        config.endpoint = \'dysmsapi.aliyuncs.com\'\r\n        self.client = Dysmsapi20170525Client(config)\r\n\r\n    def send_sms(self, phone_number: str, template_param: dict):\r\n        """\r\n        \u53d1\u9001\u77ed\u4fe1\r\n        :param phone_number: \u624b\u673a\u53f7\r\n        :param template_param: \u6a21\u677f\u53c2\u6570\r\n        :return: \u53d1\u9001\u7ed3\u679c\r\n        """\r\n        try:\r\n            send_request = SendSmsRequest(\r\n                phone_numbers=phone_number,\r\n                sign_name=settings.sms_sign_name,\r\n                template_code=settings.sms_template_code,\r\n                template_param=str(template_param)\r\n            )\r\n            response = self.client.send_sms(send_request)\r\n            return response\r\n        except Exception as e:\r\n            print(f"\u53d1\u9001\u77ed\u4fe1\u5931\u8d25: {str(e)}")\r\n            raise e\n'})})]})}function d(e={}){const{wrapper:n}={...(0,a.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(p,{...e})}):p(e)}},5658:(e,n,s)=>{s.d(n,{R:()=>i,x:()=>o});var r=s(758);const t={},a=r.createContext(t);function i(e){const n=r.useContext(a);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:i(e.components),r.createElement(a.Provider,{value:n},e.children)}}}]);