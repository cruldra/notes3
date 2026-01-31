import React from 'react';

const Disclaimer = () => {
    return (
        <div className="p-8 max-w-4xl mx-auto prose prose-slate">
            <h1 className="text-3xl font-bold mb-6 text-red-600">演示说明</h1>
            <ol className="space-y-4 list-decimal pl-5 text-lg text-slate-700">
                <li className="font-bold text-red-500">这不是最终产品效果</li>
                <li>你可以把这些交互式的演示看成是我对业务的理解，因为我在此之前并不熟悉营销这个领域</li>
                <li>并不是所有需求点都有对应的交互式演示，我只为我感到疑惑的需求点制作了演示，像“基于用户数据生成画像”这种很明确的需求点就没有演示</li>
                <li>这些演示的目的在于用可视化的方式来描述业务场景，以及降低视频或者语音会议中的沟通成本，最后就是在双方对业务达成一致的基础上，推动产品原型设计和开发工作</li>
            </ol>
        </div>
    );
};

export default Disclaimer;
