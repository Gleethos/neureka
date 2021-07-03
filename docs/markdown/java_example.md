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
     *   
     *   y.toString(): "(1):[4.0]; ->d(1):[-8.0]"    
     */
```
Matrix multiplication:
```java
    x = Tsr.of(Double.class)
            .withShape(2, 3, 1)
            .andFill(
                  3,   2, -1,
                  -2,  2,  4
            );
    y = Tsr.of(Double.class)
            .withShape(1, 3, 2)
            .andFill(
                    4, -1,  
                    3,  2,  
                    3, -1
            );
    Tsr z = Tsr.of("i0 x i1", x, y);
    
    /*
     *   z.toString(): "(2x1x2):[15.0, 2.0, 10.0, 2.0]"    
     */
```
Convolution:
```java
        x = Tsr.of(Double.class)
                .withShape(3, 3)
                .andFill(
                         1, 2, 5,
                        -1, 4,-2,
                        -2, 3, 4
                );
        y = Tsr.of(Double.class)
                .withShape(2, 2)
                .andFill(
                       -1, 3,
                        2, 3
                );
        z = Tsr.of("i0 x i1", x, y);

        // z.toString(): "(2x2):[15.0, 15.0, 18.0, 8.0)]"

        z.backward(Tsr.of(Double.class).withShape(2, 2).all(1));
        /*
         *   y.toString(): "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```

GPU execution:
```java
        Device gpu = Device.find("nvidia");
        x = Tsr.of(Double.class)
                .withShape(3, 3)
                .andFill(
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4
                )
        );
        y = Tsr.of(Double.class)
                .withShape(2, 2)
                .andFill(
                        -1, 3,
                        2, 3
                );
        gpu.store(x).store(y);        
        z = Tsr.of("i0 x i1", x, y); // <= executed on gpu!

        // z.toString(): "(2x2):[15.0, 15.0, 18.0, 8.0], "

        z.backward(Tsr.of(Double.class).withShape(2, 2).all(1));
        /*
         *   y.toString(): "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```
