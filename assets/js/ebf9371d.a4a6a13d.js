"use strict";(self.webpackChunknotes_3=self.webpackChunknotes_3||[]).push([[1392],{4127:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>l,contentTitle:()=>a,default:()=>u,frontMatter:()=>o,metadata:()=>s,toc:()=>d});const s=JSON.parse('{"id":"Python/Libraries/\u6570\u636e\u5e93/SQLModel","title":"SQLModel","description":"SQLModel\u662f\u4e00\u4e2aPython ORM\u6846\u67b6, \u5b83\u57fa\u4e8eSQLAlchemy\u548cPydantic\u6784\u5efa.\u7c7b\u4f3c\u4e8eNode\u751f\u6001\u4e2d\u7684Prisma","source":"@site/docs/Python/Libraries/\u6570\u636e\u5e93/SQLModel.mdx","sourceDirName":"Python/Libraries/\u6570\u636e\u5e93","slug":"/Python/Libraries/\u6570\u636e\u5e93/SQLModel","permalink":"/notes3/docs/Python/Libraries/\u6570\u636e\u5e93/SQLModel","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/Python/Libraries/\u6570\u636e\u5e93/SQLModel.mdx","tags":[],"version":"current","frontMatter":{},"sidebar":"python","previous":{"title":"Redis","permalink":"/notes3/docs/Python/Libraries/\u6570\u636e\u5e93/Redis"},"next":{"title":"\u81ea\u52a8\u5316","permalink":"/notes3/docs/category/\u81ea\u52a8\u5316"}}');var t=r(6070),i=r(5658);const o={},a=void 0,l={},d=[{value:"\u5b89\u88c5",id:"\u5b89\u88c5",level:2},{value:"\u793a\u4f8b",id:"\u793a\u4f8b",level:2},{value:"\u6a21\u578b\u5b9a\u4e49",id:"\u6a21\u578b\u5b9a\u4e49",level:2},{value:"\u679a\u4e3e",id:"\u679a\u4e3e",level:3},{value:"\u591a\u5bf9\u4e00",id:"\u591a\u5bf9\u4e00",level:3}];function c(e){const n={a:"a",admonition:"admonition",code:"code",h2:"h2",h3:"h3",p:"p",pre:"pre",...(0,i.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.a,{href:"https://sqlmodel.tiangolo.com/",children:"SQLModel"}),"\u662f\u4e00\u4e2a",(0,t.jsx)(n.code,{children:"Python ORM"}),"\u6846\u67b6, \u5b83\u57fa\u4e8e",(0,t.jsx)(n.code,{children:"SQLAlchemy"}),"\u548c",(0,t.jsx)(n.code,{children:"Pydantic"}),"\u6784\u5efa.\u7c7b\u4f3c\u4e8e",(0,t.jsx)(n.code,{children:"Node"}),"\u751f\u6001\u4e2d\u7684",(0,t.jsx)(n.code,{children:"Prisma"})]}),"\n",(0,t.jsx)(n.admonition,{type:"tip",children:(0,t.jsxs)(n.p,{children:["\u4f7f\u7528",(0,t.jsx)(n.a,{href:"https://prisma-client-py.readthedocs.io/en/stable/",children:"Prisma Client Python"}),"\u7684\u8bdd\u53ef\u4ee5\u7528",(0,t.jsx)(n.code,{children:"Prisma ORM"}),"\u76f8\u540c\u7684",(0,t.jsx)(n.code,{children:"API"}),"\u5728",(0,t.jsx)(n.code,{children:"Python"}),"\u4e2d\u8bbf\u95ee\u6570\u636e\u5e93,\u4f46\u8fd9\u4e2a\u5e93\u76ee\u524d\u8fd8\u4e0d\u592a\u5b8c\u5584"]})}),"\n",(0,t.jsx)(n.h2,{id:"\u5b89\u88c5",children:"\u5b89\u88c5"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:"pip install sqlmodel\n"})}),"\n",(0,t.jsx)(n.h2,{id:"\u793a\u4f8b",children:"\u793a\u4f8b"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'from typing import Optional\r\nfrom sqlmodel import Field, Session, SQLModel, create_engine, select\r\n\r\n# \u5b9a\u4e49\u6a21\u578b\r\nclass User(SQLModel, table=True):\r\n    id: Optional[int] = Field(default=None, primary_key=True)\r\n    email: str = Field(unique=True, index=True)\r\n    name: str\r\n    age: Optional[int] = None\r\n\r\nclass Post(SQLModel, table=True):\r\n    id: Optional[int] = Field(default=None, primary_key=True)\r\n    title: str\r\n    content: str\r\n    user_id: int = Field(foreign_key="user.id")\r\n\r\n# \u521b\u5efa\u6570\u636e\u5e93\u5f15\u64ce\r\nengine = create_engine("sqlite:///database.db")\r\n\r\n# \u521b\u5efa\u8868\r\nSQLModel.metadata.create_all(engine)\r\n\r\n# CRUD\u64cd\u4f5c\r\ndef create_user():\r\n    user = User(email="test@example.com", name="Test User", age=25)\r\n    with Session(engine) as session:\r\n        session.add(user)\r\n        session.commit()\r\n        session.refresh(user)\r\n    return user\r\n\r\ndef get_user(user_id: int):\r\n    with Session(engine) as session:\r\n        statement = select(User).where(User.id == user_id)\r\n        user = session.exec(statement).first()\r\n    return user\r\n\r\ndef update_user(user_id: int, name: str):\r\n    with Session(engine) as session:\r\n        statement = select(User).where(User.id == user_id)\r\n        user = session.exec(statement).first()\r\n        user.name = name\r\n        session.add(user)\r\n        session.commit()\r\n        session.refresh(user)\r\n    return user\r\n\r\ndef delete_user(user_id: int):\r\n    with Session(engine) as session:\r\n        statement = select(User).where(User.id == user_id)\r\n        user = session.exec(statement).first()\r\n        session.delete(user)\r\n        session.commit()\n'})}),"\n",(0,t.jsx)(n.h2,{id:"\u6a21\u578b\u5b9a\u4e49",children:"\u6a21\u578b\u5b9a\u4e49"}),"\n",(0,t.jsx)(n.h3,{id:"\u679a\u4e3e",children:"\u679a\u4e3e"}),"\n",(0,t.jsxs)(n.p,{children:["\u5047\u8bbe",(0,t.jsx)(n.code,{children:"\u8bfe\u7a0b\u8868"}),"\u4e2d\u6709\u4e00\u4e2a\u72b6\u6001\u5b57\u6bb5, \u5b83\u6709\u4e09\u4e2a\u503c: ",(0,t.jsx)(n.code,{children:"draft"}),", ",(0,t.jsx)(n.code,{children:"published"}),", ",(0,t.jsx)(n.code,{children:"archived"}),", \u90a3\u4e48\u6211\u4eec\u53ef\u4ee5\u8fd9\u6837\u5b9a\u4e49:"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'class CourseStatus(str, Enum):\r\n    """\r\n    \u8bfe\u7a0b\u72b6\u6001\u679a\u4e3e\r\n    """\r\n    DRAFT = "draft"\r\n    PUBLISHED = "published"\r\n    ARCHIVED = "archived"\r\n\r\n    @classmethod\r\n    def get_description(cls) -> Dict[str, str]:\r\n        return {\r\n            cls.DRAFT: "\u8349\u7a3f",\r\n            cls.PUBLISHED: "\u5df2\u53d1\u5e03",\r\n            cls.ARCHIVED: "\u5df2\u5f52\u6863",\r\n        }\r\n\r\nclass Course(SQLModel, table=True):\r\n    """\r\n    \u8bfe\u7a0b\u8868\r\n    """\r\n    id: Optional[int] = Field(default=None, primary_key=True, description="ID")\r\n    status: CourseStatus = Field(\r\n        default=CourseStatus.DRAFT,\r\n        sa_column=Column(String),\r\n        description="\u8bfe\u7a0b\u72b6\u6001"\r\n    )\n'})}),"\n",(0,t.jsx)(n.p,{children:"\u5982\u679c\u8981\u8ba9\u751f\u6210\u5217\u662f\u4e00\u4e2a\u539f\u751f\u7684\u679a\u4e3e\u7c7b\u578b(Mysql\u3001Postgresql\u7b49\u90e8\u5206\u6570\u636e\u5e93\u652f\u6301), \u53ef\u4ee5\u8fd9\u6837\u5b9a\u4e49:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'from sqlalchemy import create_engine, Column, String, Enum as SQLAlchemyEnum\r\n\r\nclass Course(SQLModel, table=True):\r\n    """\r\n    \u8bfe\u7a0b\u8868\r\n    """\r\n    id: Optional[int] = Field(default=None, primary_key=True, description="ID")\r\n    status: CourseStatus = Field(\r\n        default=CourseStatus.DRAFT,\r\n        sa_column=Column(\r\n            SQLAlchemyEnum(CourseType, name="course_type_enum", native_enum=True),\r\n            comment=\'\u8bfe\u7a0b\u7c7b\u578b\'\r\n        ),\r\n        description="\u8bfe\u7a0b\u72b6\u6001"\r\n    )\n'})}),"\n",(0,t.jsx)(n.h3,{id:"\u591a\u5bf9\u4e00",children:"\u591a\u5bf9\u4e00"}),"\n",(0,t.jsx)(n.p,{children:"\u5047\u8bbe\u8bfe\u7a0b\u8868\u548c\u5bfc\u5e08\u8868\u662f\u591a\u5bf9\u4e00,\u4e5f\u5c31\u662f\u4e00\u4e2a\u5bfc\u5e08\u53ef\u4ee5\u6709\u591a\u4e2a\u8bfe\u7a0b,\u90a3\u53ef\u4ee5\u8fd9\u6837\u5b9a\u4e49:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:'class Course(SQLModel, table=True):\r\n    """\r\n    \u8bfe\u7a0b\u8868\r\n    """\r\n    id: Optional[int] = Field(default=None, primary_key=True, description="ID")\r\n\r\n    # \u6dfb\u52a0\u5916\u952e\u5173\u8054\r\n    teacher_id: Optional[int] = Field(\r\n        default=None,\r\n        foreign_key="asm_teacher.id",\r\n        description="\u5bfc\u5e08ID"\r\n    )\r\n    # \u6dfb\u52a0\u5173\u7cfb\u5c5e\u6027\r\n    teacher: Optional["Teacher"] = Relationship(back_populates="courses")\r\n\r\nclass Teacher(SQLModel, table=True):\r\n    """\r\n    \u5bfc\u5e08\u8868\r\n    """\r\n    id: Optional[int] = Field(default=None, primary_key=True, description="ID")\r\n    # \u6dfb\u52a0\u53cd\u5411\u5173\u7cfb\r\n    courses: List["Course"] = Relationship(back_populates="teacher")\n'})})]})}function u(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(c,{...e})}):c(e)}},5658:(e,n,r)=>{r.d(n,{R:()=>o,x:()=>a});var s=r(758);const t={},i=s.createContext(t);function o(e){const n=s.useContext(i);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:o(e.components),s.createElement(i.Provider,{value:n},e.children)}}}]);