# 物联网 (IoT) 知识图谱

## 1. 感知与设备 (Perception & Devices)

### 关键组件
```mermaid
graph TD
    Device[终端设备]
    Sensor[传感器]
    Actuator[执行器]
    MCU[微控制器]
    
    Device --> Sensor
    Device --> Actuator
    Device --> MCU
```

### 参考链接

## 2. 网络与通信 (Network & Communication)

### 关键组件
```mermaid
graph TD
    Comm[通信协议]
    ShortRange[短距离: BLE/ZigBee]
    LPWAN[广域网: NB-IoT/LoRa]
    IP[IP网络: 5G/WiFi]
    
    Comm --> ShortRange
    Comm --> LPWAN
    Comm --> IP
```

### 参考链接

## 3. 边缘计算 (Edge Computing)

**eKuiper**: LF Edge eKuiper 是一款专为资源受限的边缘设备设计的轻量级物联网数据分析和流处理引擎。

### 关键组件
```mermaid
graph TD
    Edge[边缘计算]
    eKuiper[LF Edge eKuiper]
    Gateway[边缘网关]
    Analysis[实时分析]
    
    Edge --> Gateway
    Gateway --> eKuiper
    eKuiper --> Analysis
```

### 参考链接
- [LF Edge eKuiper 官网](https://ekuiper.org/)

## 4. 平台与云 (Platform & Cloud)

### 关键组件
```mermaid
graph TD
    Platform[IoT 平台]
    DeviceMgmt[设备管理]
    DataProcess[数据处理]
    RuleEngine[规则引擎]
    
    Platform --> DeviceMgmt
    Platform --> DataProcess
    Platform --> RuleEngine
```

### 参考链接

## 5. 行业应用 (Applications)

### 关键组件
```mermaid
graph TD
    App[行业应用]
    SmartHome[智能家居]
    SmartCity[智慧城市]
    IIoT[工业互联网]
    IoV[车联网]
    
    App --> SmartHome
    App --> SmartCity
    App --> IIoT
    App --> IoV
```

### 参考链接

## 6. 安全与标准 (Security & Standards)

### 关键组件
```mermaid
graph TD
    Security[安全体系]
    Auth[身份认证]
    Encrypt[数据加密]
    Audit[安全审计]
    Standard[行业标准]
    
    Security --> Auth
    Security --> Encrypt
    Security --> Audit
    Security --> Standard
```

### 参考链接
