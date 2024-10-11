#!/bin/bash

# 启动 Tornado 服务器
# 手动将 Forecast 目录添加到 PYTHONPATH 中。这样，Python 就能够找到 darts.preprocess.custom.process 模块。
export PYTHONPATH=$PYTHONPATH:/home/project/EMS/load_modify/Forecast
python3 setup.py &

# 给 Tornado 服务一点时间启动
sleep 10

# 定义一个数组，包含所有的 JSON 数据
json_data_list=(
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2023-10-05 09:50:34",
#    "taskType": 1,
#    "mode": 1,
#    "startTime": "2023-09-05 05:00:00",
#    "endTime": "2023-09-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "广德东威科技0.8MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower": "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2023-10-05 09:50:34",
#    "taskType": 1,
#    "mode": 24,
#    "startTime": "2023-09-05 05:00:00",
#    "endTime": "2023-09-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "广德东威科技0.8MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower": "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 1,
#    "mode": 1,
#    "startTime": "2023-09-05 05:00:00",
#    "endTime": "2023-09-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "shanghai",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "85000",
#        "loadRatedPower": "2635"
#    }
#}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "shanghai",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "85000",
        "loadRatedPower": "2635"
    }
}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 1,
#    "mode": 1,
#    "startTime": "2023-09-05 05:00:00",
#    "endTime": "2023-09-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "固德威（广德）2.8MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "2800000",
#        "loadRatedPower": "2635"
#    }
#}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "固德威（广德）2.8MW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "2800000",
        "loadRatedPower": "2635"
    }
}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 1,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "广德苏农生物科技440kW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "440000",
        "loadRatedPower": "2635"
    }
}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "广德苏农生物科技440kW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "440000",
        "loadRatedPower": "2635"
    }
}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 1,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "安徽未来饰界440KW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "440000",
        "loadRatedPower": "2635"
    }
}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "安徽未来饰界440KW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "440000",
        "loadRatedPower": "2635"
    }
}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 1,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "广德亿盛精密科技320KW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "320000",
        "loadRatedPower": "2635"
    }
}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "广德亿盛精密科技320KW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "320000",
        "loadRatedPower": "2635"
    }
}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 1,
#    "mode": 1,
#    "startTime": "2023-09-05 05:00:00",
#    "endTime": "2023-09-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "安庆经开区孵化园一期400kw分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "400000",
#        "loadRatedPower": "2635"
#    }
#}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "安庆经开区孵化园一期400kw分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "400000",
        "loadRatedPower": "2635"
    }
}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 1,
#    "mode": 1,
#    "startTime": "2023-09-05 05:00:00",
#    "endTime": "2023-09-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "苏州路之遥科技股份有限公司0.77MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "770000",
#        "loadRatedPower": "2635"
#    }
#}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "苏州路之遥科技股份有限公司0.77MW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "770000",
        "loadRatedPower": "2635"
    }
}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 1,
#    "mode": 1,
#    "startTime": "2023-09-05 05:00:00",
#    "endTime": "2023-09-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "苏州现代工业坊1栋496.1kw分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "496100",
#        "loadRatedPower": "2635"
#    }
#}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "苏州现代工业坊1栋496.1kw分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "496100",
        "loadRatedPower": "2635"
    }
}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 1,
#    "mode": 1,
#    "startTime": "2023-09-05 05:00:00",
#    "endTime": "2023-09-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "宣城安徽久泰新材料科技有限公司0.462MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "462000",
#        "loadRatedPower": "2635"
#    }
#}'
'{
    "calculateCode": "202401011149321002",
    "createTime": "2024-10-05 09:50:34",
    "taskType": 1,
    "mode": 24,
    "startTime": "2023-09-05 05:00:00",
    "endTime": "2023-09-06 04:45:00",
    "frequency": "15",
    "stationInfo": {
        "id": "宣城安徽久泰新材料科技有限公司0.462MW分布式光伏",
        "longitude": "119.45776",
        "latitude": "30.88567",
        "pvRatedPower": "462000",
        "loadRatedPower": "2635"
    }
}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 1,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "安徽未来饰界440KW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 24,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "安徽未来饰界440KW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 1,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "固德威（广德）2.8MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 24,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "固德威（广德）2.8MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 1,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "广德东威科技0.8MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 24,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "广德东威科技0.8MW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 1,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "广德苏农生物科技440kW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 24,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "广德苏农生物科技440kW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 1,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "广德亿盛精密科技320KW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 24,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "广德亿盛精密科技320KW分布式光伏",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 1,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "力恒动力",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
#'{
#    "calculateCode": "202401011149321002",
#    "createTime": "2024-10-05 09:50:34",
#    "taskType": 0,
#    "mode": 24,
#    "startTime": "2023-05-05 05:00:00",
#    "endTime": "2023-05-06 04:45:00",
#    "frequency": "15",
#    "stationInfo": {
#        "id": "力恒动力",
#        "longitude": "119.45776",
#        "latitude": "30.88567",
#        "pvRatedPower": "800000",
#        "loadRatedPower":  "2635"
#    }
#}'
)

# 循环遍历每个 JSON 数据并 POST 到 /Train 端点   192.168.181.150
for json_data in "${json_data_list[@]}"; do
    echo "Training model with current dataset..."
    curl -s -X POST http://localhost:40088/Train \
        -H "Content-Type: application/json" \
        -d "$json_data"

    # 等待一段时间以确保不会太快连续发起请求
    sleep 5
done

# 保持服务运行（如果需要）
wait