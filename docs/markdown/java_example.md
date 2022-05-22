# Neureka with Java #

Simple scalar calculation:
```java
    Tsr<Double> x = Tsr.of(3).setRqsGradient(true);
    Tsr<Double> b = Tsr.of(-4);
    Tsr<Double> w = Tsr.of(2);
        
    Tsr<Double> y = Tsr.of("((i0 + i1) * i2) ^ 2", x, b, w);
    
    /*
     *   f(x) = ((x-4)*2)^2; :=>  f(3) = 4
     *   f(x)' = 8*x - 32 ;  :=>  f(3)' = -8
     */
     
     y.backward();
     
     System.out.println(x); // "(1):[3.0]:g:[-8.0]"
     // Here '-8' is the derivative as well as the gradient of x!
     
```
Matrix multiplication:
```java
    var x = Tsr.of(Double.class)
                    .withShape(2, 3)
                    .andFill(
                          3.0,   2.0, -1.0,
                          -2.0,  2.0,  4.0
                    );
                    
    var y = Tsr.of(Double.class)
                .withShape(3, 2)
                .andFill(
                        4.0, -1.0,  
                        3.0,  2.0,  
                        3.0, -1.0
                );
            
    Tsr<Double> z = x.matMul(y);
    
    System.out.println(z); 
    /*
        (2x2):[
           [  15.0 ,   2.0  ],
           [  10.0 ,   2.0  ]
        ]
    */
```
Convolution:
```java
        var x = Tsr.of(Double.class)
                    .withShape(3, 3)
                    .andFill(
                            1.0, 2.0, 5.0,
                            -1.0, 4.0,-2.0,
                            -2.0, 3.0, 4.0
                    );
                    
        var y = Tsr.of(Double.class)
                    .withShape(2, 2)
                    .andFill(
                            -1.0, 3.0,
                            2.0, 3.0
                    );

        y.setRqsGradient(true);

        var z = Tsr.of("i0 x i1", x, y);

        System.out.println(z); // "(2x2):[15.0, 15.0, 18.0, 8.0)]"

        z.backward(Tsr.of(Double.class).withShape(2, 2).all(1.0));

        System.out.println(y);
        /*
            (2x2):[
               [  -1.0 ,   3.0  ],
               [   2.0 ,   3.0  ]
            ]
            :g:[6.0, 9.0, 4.0, 9.0]
         */
```

GPU execution:
```java
        Device gpu = Device.find("nvidia").orElse(CPU.get());
        var x = Tsr.of(Double.class)
                    .withShape(3, 3)
                    .andFill(
                             1.0,  2.0,  5.0,
                            -1.0,  4.0, -2.0,
                            -2.0,  3.0,  4.0
                    )
        );
        var y = Tsr.of(Double.class)
                    .withShape(2, 2)
                    .andFill(
                            -1.0, 3.0,
                             2.0, 3.0
                    );
                    
        gpu.store(x).store(y);   
        
        var z = Tsr.of("i0 x i1", x, y); // <= executed on gpu!

        System.out.println(z); // "(2x2):[15.0, 15.0, 18.0, 8.0], "

        z.backward(Tsr.of(Double.class).withShape(2, 2).all(1.0));
        /*
            "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```
