# gpu-redis-aggregator
gpu-redis-aggregator
可以先下载deploy文件夹里面的所有yaml到本地，然后直接运行即可
启动命令是：先启动所有yaml文件：kubectl apply -f .(需要deployment.yaml, rbac.yaml  , service.yaml )
连接端口：kubectl -n gpu-mig-monitoring port-forward svc/gpu-redis-aggregator 8080:8080
通过端口进行查询：curl http://127.0.0.1:8080/report
后续直接通过deployment.yaml文件中的镜像地址来进行更新版本即可
如果是其他的服务中需要访问，直接进行即可：curl http://gpu-redis-aggregator.gpu-mig-monitoring.svc.cluster.local:8080/report
