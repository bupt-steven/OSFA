<?php
     /*
     //配置文件区
     define(nodeMaxNum, 16);
     define(nodePortNum,4);
     define(nodeLineNum,nodePortNum/2);
     //信息区
     //算法思想：每次根据端口关系，求出每个节点的互联关系。然后求出每个路径的图
     $linkInfoAll[nodeMaxNum][nodeMaxNum];//全局端口关系
     $linkFlowAll[nodeMaxNum][nodeMaxNum];//全局互联关系*/
     
     //目前根据端口先默认fatTree,拓扑结果为一个环
     function initial($line = 8) {
         $topoStr = file_get_contents("topo.txt",true);
         $topoArray = explode("\n", $topoStr);
         foreach($topoArray as $key => $value) {
             $valueArray = explode(" ",$value);
             if ($key == $line){
                 break;
             }
             for($i = 0; $i <count($valueArray); $i++){
                 $valueArray[$i] = (int)$valueArray[$i];
             } 
             $topoOut[$key] = $valueArray; 
         }
         return $topoOut;         
     }
     function Dijkstra($topo,$v=0) {
         $maxInt = 100000;
         $dist = array();//纪录最短距离
         $pre  = array();//纪录最段距离的上一节点
         $queue = array();
         for($i = 0; $i < count($topo); $i++ ) {
             for($j=0; $j < count($topo); $j++) {
                 if($j != $i && $topo[$i][$j] == 0 )
                     $topo[$i][$j] =  $maxInt;
             }
         }
         for($i = 0 ; $i < count($topo); $i++) {
             $dist[$i] = $topo[$v][$i];
             $queue[$i] = false;
             if($dist[$i] == $maxInt)
                 $pre[$i][0] = -1;
             else
                 $pre[$i][0] = $v;
         }
         $dist[$v] = 0;
         $queue[$v] = true;
         for($i = 1; $i < count($topo); $i++) {
             $mindist = $maxInt;//当前最小值
             $u = $v;//前驱节点
             for($j=0; $j <count($topo); $j++) {
                 if( !$queue[$j] && $dist[$j] < $mindist) {
                     $u = $j;                             // u保存当前邻接点中距离最小的点的号码 
                     $mindist = $dist[$j];
                 }

             }
             $queue[$u] = true; 
             for($j = 0; $j < count($topo);$j++) {
                 if( !$queue[$j] && $topo[$u][$j] < $maxInt) {
                     if($dist[$u] + $topo[$u][$j] < $dist[$j])     //在通过新加入的u点路径找到离v0点更短的路径  
                     {
                         $dist[$j] = $dist[$u] + $topo[$u][$j];    //更新dist 
                         $pre[$j][0] = $u;                    //记录前驱顶点 
                     }
                     if($dist[$u] + $topo[$u][$j] == $dist[$j])
                     {
                         if ($pre[$j][0] == -1)
                             $pre[$j][0] = $u;
                         else if (!in_array($u, $pre[$j]) )//去重
                             $pre[$j][] = $u;
                     }
                 }
             }
          }
             //var_dump($pre);
             return $pre;
     }
     $result = [];// 最短路径结果输出
     $stack= array();//辅助栈
     function dfs( $pre,$targetNode,$v) {
         //return 一个数组，每一列记录一条最短路径   
         global $stack;
         global $result;
         for($i = 0; $i < count($pre[$targetNode]); $i++) {
             array_push($stack,$pre[$targetNode][$i]);
             if($pre[$targetNode][$i] == $v) {
                 $result[] = $stack;
                 array_pop($stack);
                 return ;
             }
             else {
                 dfs($pre,$pre[$targetNode][$i],$v);
             }
             array_pop($stack);
         }
     } 
     $egdeHashInfo = [];
     //边信息，每条边有一个唯一id，值为字符串（1，2），代表1连接2
     $taskInfo = [];
     //每个任务，key值为taskid，value一个数组，其中value［0］如（0，4），起始节点，value［1］为task的流大小
     $shortRouter = [];
     //最短路径记录。key值为taskid，value为一个数组，每一值代表一个路径；
     $flowInfo = [];
     //将flow映射到每条边上，key值为flowid，value为egdeid
     $edgeFlowInfo = [];
     //记录每条边需要传的flowid，以及大小
     $edgeTcpLimitInfo = [];
     //记录每条边的Tcp阀值
     $tm  = 1 ;
     //默认TCP的一跳时间
     $T = rand(0,24);
     //默认一个时间段T内不会有发生拓扑改变。

     function edgeHash ($topo) {
         $outPut = [];
         for ($i = 0; $i <count($topo); $i++) {
             for($j =0 ;$j < $i ; $j++) {
                 if($topo[$i][$j] == 1) { 
                     $outPut[] = "$i".','."$j";
                 }
             }
         }
         return $outPut;
     }
     
     function taskToflow ($task,$topo) {
         global $result;

         $shortRouter = [];
         //最短路径记录。key值为taskid，value为一个数组，每一值代表一个路径；

         for($i = 0 ;$i < count($task); $i++) {
             $result = [];
             $fromTo = explode(",",$task[$i][0]);
             $preArray = Dijkstra($topo,(int)$fromTo[0]);
             dfs($preArray,$fromTo[1],(int)$fromTo[0]);
             $shortRouter[$i] = $result; 
             for($t = 0 ;$t < count($shortRouter[$i]);$t++) {
                 array_unshift($shortRouter[$i][$t], (int)$fromTo[1]);
             }
         }
         return $shortRouter;
     } 
     function getFlowId($shortRouter){
          $flowInfo = [];
          foreach ($shortRouter as $val) {
              for($i = 0; $i < count($val); $i++) {
                  $flowInfo[] = $val[$i]; 
              }
              //$flowInfo[]
          }
          return $flowInfo;
     }

     function getFlowNum($shortRouter, $task){
          $flowNumInfo = [];
          foreach($shortRouter as $key => $val) {
              for($i = 0; $i < count($val); $i++) 
                  $flowNumInfo[] = $task[$key][1]/count($val);
          }
          return $flowNumInfo;
     }
    

     function flowToEdge($flowInfo, $egdeHashInfo, $flowNumInfo) {
         $edgeFlowInfo = [];
         foreach ($flowInfo as $key => $val) {
             for($i = 0; $i < count($val)-1; $i++ ) {
                 if ($val[$i] > $val[$i+1])
                     $tempData = $val[$i].','.$val[$i+1];
                 else
                     $tempData = $val[$i+1].','.$val[$i];
                 
                 if (in_array($tempData,$egdeHashInfo)){
                     $k = array_search($tempData, $egdeHashInfo);
                     if (!isset($edgeFlowInfo[$k])){
                         $edgeFlowInfo[$k]['value'] = $flowNumInfo[$key];
                         $edgeFlowInfo[$k]['flowid'] = "$key";
                     }
                     else {
                         $edgeFlowInfo[$k]['value'] += $flowNumInfo[$key];
                         $edgeFlowInfo[$k]['flowid'] .= ",$key";
                     }
                 }
             }
         }
         return $edgeFlowInfo;
     }
     function intitialTcpWindow($flowNumInfo){
         //flowNumInfo的key为每条流的id
         //正式系统中每次初始化后启动窗口值为随机值，而非现在每次启动。
         $outPut = [];
         for($i =0 ;$i <count($flowNumInfo); $i++) {
             $outPut [] = $i+1;
         }
         return $outPut;
     }
     $topo = initial(8);
     //@ini_set('memory_limit','4G');
     $egdeHashInfo = edgeHash($topo);
     $task = array(
          "0" => array("0,4",12),
          "1" => array("0,4",12),
      );
     //寻找最短路径
     $pre = Dijkstra($topo,0);
     dfs($pre,4,0);
     var_dump($result);exit();
     $shortRouter = taskToflow($task, $topo);
     $flowInfo = getFlowId($shortRouter);
     $flowNumInfo = getFlowNum($shortRouter,$task);
     $edgeFlowInfo =  flowToEdge($flowInfo,$egdeHashInfo, $flowNumInfo);
     $TcpWindow = intitialTcpWindow($flowNumInfo);
     for($curtime = 0 ;$curtime <= 2; $curtime++) {
         //正式系统中每次初始化后启动窗口值为随机值，而非现在每次启动。
         //todo: 如何获得超时呢？
         for($i = 0; $i < count($TcpWindow); $i++){
            if ($TcpWindow[$i] < 8)
               $TcpWindow[$i] = $TcpWindow[$i]*2;
            else 
               $TcpWindow[$i] = $TcpWindow[$i]+1;
         }//先让其增大，然后再判断是否拥塞
         foreach ($edgeFlowInfo as $val) {
             $tranSum =  10000;//保证进循环
             while ($tranSum > 20) {
                 $flowArray = explode(",", $val['flowid']);
                 $tranSum =0;
                 for($j =0 ; $j <count($flowArray); $j++) {
                     $tranSum += (int)$TcpWindow[$flowArray[$j]]*1;
                 }
                 if ($tranSum <= 20) {
                     break;
                 } else {
                     $decreaseFlow = rand(0,count($flowArray)-1);
                     $TcpWindow[$flowArray[$decreaseFlow]] = $TcpWindow[$flowArray[$decreaseFlow]]/2;
                     echo $decreaseFlow."你被减速了"."\n";
                 }
             }
         }
         echo "--------------------------------------------------------------------------\n";
         var_dump($TcpWindow);
         echo "--------------------------------------------------------------------------\n";
         //此时算完了当前时刻无拥塞下的每条流的tcp窗口
 
         for($index = 0 ;$index < count($flowNumInfo); $index++) {
             if($flowNumInfo[$index] > ($TcpWindow[$index]*1))
                 $flowNumInfo[$index] = $flowNumInfo[$index] - ($TcpWindow[$index]*1);
             else {//被传完了
                 $flowNumInfo[$index] = 0;
                 foreach($edgeFlowInfo as $key => $value){//有问题
                     //var_dump($edgeFlowInfo);
                         $clearArray = explode(",", $value['flowid']);
                         if (in_array($index, $clearArray)) {
                             $needDelete = array_search($index, $clearArray);
                             echo $clearArray[$needDelete]." 被传完了,时间是 $curtime 现在的边是".$key."\n";
                             unset($clearArray[$needDelete]); 
                             if(count($clearArray) == 0) {
                                 unset($edgeFlowInfo[$key]);
                             } else {
                                 $clearArrayStr = implode(",", $clearArray);
                                 $edgeFlowInfo[$key]['flowid'] = $clearArrayStr;
                             }
                         }
                 }
                 /*for($m = 0; $m < count($TcpWindow); $m++) {
                     if (isset($TcpWindow[$index])) {
                         unset($TcpWindow[$index]);
                     }
                 }*/

             }
             var_dump($flowNumInfo);
             //做个跳出判断，完事
         }
         //var_dump($TcpWindow);
         //var_dump($flowNumInfo);
     }
     //var_dump($shortRouter);
     //var_dump($flowNumInfo);
     //var_dump($edgeFlowInfo);
     //var_dump($TcpWindow);
     
?>
