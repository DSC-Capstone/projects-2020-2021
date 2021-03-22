// const { exec } = require("child_process");

// exec("ls -la", (error, stdout, stderr) => {
//     if (error) {
//         console.log(`error: ${error.message}`);
//         return;
//     }
//     if (stderr) {
//         console.log(`stderr: ${stderr}`);
//         return;
//     }
//     console.log(`stdout: ${stdout}`);
// });


var speed = 0.0;
function send_command(button_id,content) {
var textFile = null,
  makeTextFile = function (text) {
    var data = new Blob([text], {type: 'text/plain'});

    // If we are replacing a previously generated file we need to
    // manually revoke the object URL to avoid memory leaks.
    if (textFile !== null) {
      window.URL.revokeObjectURL(textFile);
    }

    textFile = window.URL.createObjectURL(data);

    return textFile;
  };


  var create = document.getElementById(button_id);

  create.addEventListener('click', function () {
    var link = document.createElement('a');
    link.setAttribute('download', 'command.txt');
    link.href = makeTextFile(content);
    document.body.appendChild(link);

    // wait for the link to be added to the document
    window.requestAnimationFrame(function () {
      var event = new MouseEvent('click');
      link.dispatchEvent(event);
      document.body.removeChild(link);
    });
    
  }, false);
}



function refresh(node)
{
   var times = 100; // gap in Milli Seconds;

   (function startRefresh()
   {
      var address;
      if(node.src.indexOf('?')>-1)
       address = node.src.split('?')[0];
      else 
       address = node.src;
      node.src = address+"?time="+new Date().getTime();

      setTimeout(startRefresh,times);
   })();

}

window.onload = function()
{
  // var node = document.getElementById('img_camera');
  // refresh(node);
  var node2 = document.getElementById('img_navi');
  refresh(node2);
  // you can refresh as many images you want just repeat above steps
}
let vueApp = new Vue({
    el: "#vueApp",
    data: {
        // ros connection
        ros: null,
        rosbridge_address: 'ws://localhost:9090',
        connected: false,
        // page content
        menu_title: 'Connect to RosBridge',
        sub_title: 'RosBridge Subscribe',
        pub_title: 'Navigate Robot',
        camera_title: "Input from Camera",
        IMU_title: "Input from IMU",
        navi_title: "RRT Nagivation",
        IMU_G1_title: "IMU Viz 1",
        bat_title:"Robot Speed& Battery Chart",
        speed_title:" Orientation angle of Robot"
    },
    methods: {
        connect: function() {
            // define ROSBridge connection object
            this.ros = new ROSLIB.Ros({
                url: this.rosbridge_address
            })

            // define callbacks
            this.ros.on('connection', () => {

                this.connected = true
                console.log('Connection to ROSBridge established!')

            // subsribe to topic
            let topic = new ROSLIB.Topic({
                ros: this.ros,
                name: '/cmd_vel',
                messageType: 'geometry_msgs/Twist'
            })
            topic.subscribe((message) => {
                speed = Math.round(parseFloat(message.linear.x)*100);
                console.log(speed)
            })
            let topic1 = new ROSLIB.Topic({
                ros: this.ros,
                name: '/car_1/base/odom',
                messageType: 'nav_msgs/Odometry'
            })
            topic1.subscribe((message) => {
                document.getElementById("position").innerHTML = "x: " +message.pose.pose.position.x+" y: " +message.pose.pose.position.y+" z: " +message.pose.pose.position.z;
                document.getElementById("orientation").innerHTML = "x: " +message.pose.pose.orientation.x+" y: " +message.pose.pose.orientation.y+" z: " +message.pose.pose.orientation.z;
                document.getElementById("angular").innerHTML = "x: " +message.twist.twist.angular.x+" y: " +message.twist.twist.angular.y+" z: " +message.twist.twist.angular.z;
                document.getElementById("linear").innerHTML = "x: " +message.twist.twist.linear.x+" y: " +message.twist.twist.linear.y+" z: " +message.twist.twist.linear.z;
            })

            // let topic0 = new ROSLIB.Topic({
            //     ros: this.ros,
            //     name: '/car_1/base/odom',
            //     messageType: 'nav_msgs/Odometry'
            // })
            // topic0.subscribe((message) => {
            //     document.getElementById("mappos").value +=""
                
            // })

            let topic0 = new ROSLIB.Topic({
                ros: this.ros,
                name: '/tf',
                messageType: 'tf2_msgs/TFMessage'
            })
            topic0.subscribe((message) => {
		if (parseFloat(message.transforms[0].transform.translation.x) != 0) {
		        document.getElementById("mappos").value +="                              X :" + parseFloat(message.transforms[0].transform.translation.x).toFixed(11)  
		        document.getElementById("mappos").value +="                                           Y :" + parseFloat(message.transforms[0].transform.translation.y).toFixed(11)
		        document.getElementById("mappos").value +="                                           Z :" + parseFloat(message.transforms[0].transform.translation.z).toFixed(11) + "\n"
		}
            })



            let topic2 = new ROSLIB.Topic({
                ros: this.ros,
                name: '/imu',
                messageType: 'sensor_msgs/Imu'
            })
            topic2.subscribe((message) => {
                // console.log(message)

                document.getElementById("linear_acceleration").innerHTML = "x: " +message.linear_acceleration.x+" y: " +message.linear_acceleration.y+" z: " +message.linear_acceleration.z;
                document.getElementById("orientation_imu").innerHTML = "x: " +message.orientation.x+" y: " +message.orientation.y+" z: " +message.orientation.z+" w: " +message.orientation.w;
                document.getElementById("angular_velocity").innerHTML = "x: " +message.angular_velocity.x+" y: " +message.angular_velocity.y+" z: " +message.angular_velocity.z;
                const quaternion = new THREE.Quaternion(message.angular_velocity.x,message.angular_velocity.y,message.angular_velocity.z,message.angular_velocity.w)
                const angle = new THREE.Euler()
                angle.setFromQuaternion(quaternion)
                document.getElementById("angle").innerHTML = "Roll "+angle._x+ " Pitch "+ angle._y+" Yaw "+angle._z

            })

            })
            this.ros.on('error', (error) => {
                console.log('Something went wrong when trying to connect')
                console.log(error)
            })
            this.ros.on('close', () => {
                this.connected = false
                console.log('Connection to ROSBridge was closed!')
            })
        },
        disconnect: function() {
            this.ros.close()
        },
        publish: function() {
            let topic = new ROSLIB.Topic({
                ros: this.ros,
                name: '/move_base_simple/goal',
                messageType: 'geometry_msgs/PoseStamped'
            })
            var positionx= document.getElementById("positionx").value;
            var positiony= document.getElementById("positiony").value;
            var orientationz= document.getElementById("orientationz").value;
            var orientationw= document.getElementById("orientationw").value;
            positionx=parseFloat(positionx);
            positiony=parseFloat(positiony);
            orientationz=parseFloat(orientationz);
            orientationw=parseFloat(orientationw);

            let message = new ROSLIB.Message({
              
            header: {stamp: 'dummy', frame_id: "odom"}, 
            pose: {
              position: {x:positionx , y: positiony, z: 0.0}, 
              orientation: {x: 0.0, y: 0.0, z: orientationz, w: orientationw}}
            })
            topic.publish(message)
            console.log(message)
        },

        turnRight: function() {
            let topic = new ROSLIB.Topic({
                ros: this.ros,
                name: '/move_base_simple/goal',
                messageType: 'geometry_msgs/PoseStamped'
            })
            let message = new ROSLIB.Message({
                linear: { x: 1, y: 0, z: 0, },
                angular: { x: 0, y: 0, z: -2, },
            })
            topic.publish(message)
        },

        preset: function() {
          document.getElementById("positionx").value = "-2.51826405525";
          document.getElementById("positiony").value = "-0.391522318125";
          document.getElementById("orientationz").value = "0.958047977335";
          document.getElementById("orientationw").value = "-0.286607873451";
            
        },
        stop: function() {
            let topic = new ROSLIB.Topic({
                ros: this.ros,
                name: '/turtle1/cmd_vel',
                messageType: 'geometry_msgs/Twist'
            })
            let message = new ROSLIB.Message({
                linear: { x: 0, y: 0, z: 0, },
                angular: { x: 0, y: 0, z: 0, },
            })
            topic.publish(message)
        },
    },
    mounted() {
        // page is ready
        console.log('page is ready!')
    },
})


var contain=Highcharts.chart('container', {

  chart: {
    polar: true
  },

  title: {
    text: ''
  },

  credits: {
    enabled: false
  },

  subtitle: {
    text: 'IMU Angular Twist'
  },

  pane: {
    startAngle: 0,
    endAngle: 360
  },

  xAxis: {
    tickInterval: 45,
    min: 0,
    max: 360,
    labels: {
      format: '{value}'
    }
  },

  yAxis: {
    min: 0
  },

  plotOptions: {
    series: {
      pointStart: 0,
      pointInterval: 45
    },
    column: {
      pointPadding: 0,
      groupPadding: 0
    }
  },
  series: [{
    type: 'column',
    name: 'Roll',
    data: [1, 1, 8, 1, 1, 1, 1, 1]
  }, {
    type: 'line',
    name: 'YPitch',
    data: [1,1,1,1, 1, 1, 8, 1]
  }, {
    type: 'area',
    name: 'Yaw',
    data: [1, 8, 1, 1, 1, 1, 1, 1]
  }]


});
var gaugeOptions = {
    chart: {
        type: 'solidgauge'
    },

    title: null,

    pane: {
        center: ['50%', '85%'],
        size: '100%',
        startAngle: -90,
        endAngle: 90,
        background: {
            backgroundColor:
                Highcharts.defaultOptions.legend.backgroundColor || '#EEE',
            innerRadius: '60%',
            outerRadius: '100%',
            shape: 'arc'
        }
    },

    exporting: {
        enabled: false
    },

    tooltip: {
        enabled: false
    },

    // the value axis
    yAxis: {
        stops: [
            [0.1, '#55BF3B'], // green
            [0.5, '#DDDF0D'], // yellow
            [0.9, '#DF5353'] // red
        ],
        lineWidth: 0,
        tickWidth: 0,
        minorTickInterval: null,
        tickAmount: 2,
        title: {
            y: -70
        },
        labels: {
            y: 16
        }
    },

    plotOptions: {
        solidgauge: {
            dataLabels: {
                y: 5,
                borderWidth: 0,
                useHTML: true
            }
        }
    }
};



var gaugeOptions2 = {
    chart: {
        type: 'solidgauge'
    },

    title: null,

    pane: {
        center: ['50%', '85%'],
        size: '100%',
        startAngle: -90,
        endAngle: 90,
        background: {
            backgroundColor:
                Highcharts.defaultOptions.legend.backgroundColor || '#EEE',
            innerRadius: '60%',
            outerRadius: '100%',
            shape: 'arc'
        }
    },

    exporting: {
        enabled: false
    },

    tooltip: {
        enabled: false
    },

    // the value axis
    yAxis: {
        stops: [
            [0.3,'#DF5353' ], // green
            [0.5, '#DDDF0D'], // yellow
            [0.9, '#55BF3B' ] // red
        ],
        lineWidth: 0,
        tickWidth: 0,
        minorTickInterval: null,
        tickAmount: 2,
        title: {
            y: -70
        },
        labels: {
            y: 16
        }
    },

    plotOptions: {
        solidgauge: {
            dataLabels: {
                y: 5,
                borderWidth: 0,
                useHTML: true
            }
        }
    }
};
// The speed gauge
var chartSpeed = Highcharts.chart('container-speed', Highcharts.merge(gaugeOptions, {
    yAxis: {
        min: 0,
        max: 70,
 
        title: {
            text: 'Speed'
        }

    },

    credits: {
        enabled: false
    },

    series: [{
        name: 'Speed',
        data: [speed],
        dataLabels: {
            format:
                '<div style="text-align:center">' +
                '<span style="font-size:25px">{y}</span><br/>' +
                '<span style="font-size:12px;opacity:0.4">m/s</span>' +
                '</div>'
        },
        tooltip: {
            valueSuffix: ' m/s'
        }
    }]

}));

// The RPM gauge
var chartRpm = Highcharts.chart('container-bat', Highcharts.merge(gaugeOptions2, {
    yAxis: {
        min: 0,
        max: 100,
        title: {
            text: 'Battery'
        }
    },

    credits: {
        enabled: false
    },

    series: [{
        name: 'Battery',
        data: [50],
        dataLabels: {
            format:
                '<div style="text-align:center">' +
                '<span style="font-size:25px">{y:.1f}</span><br/>' +
                '<span style="font-size:12px;opacity:0.4">' +
                '%' +
                '</span>' +
                '</div>'
        },
        tooltip: {
            valueSuffix: '%'
        }
    }]

}));

// Bring life to the dials
setInterval(function () {
    // Speed
    var point,
        newVal,
        inc;

    if (chartSpeed) {
        point = chartSpeed.series[0].points[0];
        // console.log(chartSpeed.series[0].points)
        point.update(speed);
    }

      
    // RPM
    if (chartRpm) {
        point = chartRpm.series[0].points[0];
        inc = Math.round((Math.random() - 0.5) * 100);
        newVal = point.y + inc;

        if (newVal < 0 || newVal > 100) {
            newVal = 50
        }

        point.update(newVal);
    }

    if (contain) {
      for (var k=0; k<3; k++){
        for(var i=0; i<8; i++){
          point = contain.series[k].points[i];
          newVal=point.y*0+1;
          point.update(newVal)

        }
        var num= Math.round(Math.random()*8)
        point = contain.series[k].points[num];
        newVal=point.y*0+8;
        point.update(newVal)
        // filler[]=8
        // point.update(filler);
      }
    }



}, 2000);

series: [{
    type: 'column',
    name: 'Roll',
    data: [1, 1, 8, 1, 1, 1, 1, 1]
  }, {
    type: 'line',
    name: 'YPitch',
    data: [1,1,1,1, 1, 1, 8, 1]
  }, {
    type: 'area',
    name: 'Yaw',
    data: [1, 8, 1, 1, 1, 1, 1, 1]
  }]


Highcharts.chart('container-angle', {

    chart: {
        type: 'gauge',
        plotBackgroundColor: null,
        plotBackgroundImage: null,
        plotBorderWidth: 0,
        plotShadow: false
    },

    title: {
        text: 'Orientation Angle'
    },

    credits: {
        enabled: false
    },

    pane: {
        startAngle: -150,
        endAngle: 150,
        background: [{
            backgroundColor: {
                linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                stops: [
                    [0, '#FFF'],
                    [1, '#333']
                ]
            },
            borderWidth: 0,
            outerRadius: '109%'
        }, {
            backgroundColor: {
                linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
                stops: [
                    [0, '#333'],
                    [1, '#FFF']
                ]
            },
            borderWidth: 1,
            outerRadius: '107%'
        }, {
            // default background
        }, {
            backgroundColor: '#DDD',
            borderWidth: 0,
            outerRadius: '105%',
            innerRadius: '103%'
        }]
    },

    // the value axis
    yAxis: {
        min: -180,
        max: 180,

        minorTickInterval: 'auto',
        minorTickWidth: 1,
        minorTickLength: 10,
        minorTickPosition: 'inside',
        minorTickColor: '#666',

        tickPixelInterval: 30,
        tickWidth: 2,
        tickPosition: 'inside',
        tickLength: 10,
        tickColor: '#666',
        labels: {
            step: 2,
            rotation: 'auto'
        },
        title: {
            text: ''
        },
        plotBands: [{
            from: -200,
            to: 200,
            color: '#55BF3B' // green
        }]
    },

    series: [{
        name: 'Speed',
        data: [0],
        tooltip: {
            valueSuffix: ' degree'
        }
    }]

},


// Add some life
function (chart) {
    if (!chart.renderer.forExport) {
        setInterval(function () {
            var point = chart.series[0].points[0],
                newVal,
                inc = Math.round((Math.random() - 0.5) * 100);

            newVal = point.y + inc;
            if (newVal < -180 || newVal > 200) {
                newVal = point.y - inc;
            }

            point.update(newVal);

        }, 3000);
    }
});
