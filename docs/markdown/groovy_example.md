# Neureka with Groovy #

Simple scalar calculation:
```java
    def x = new Tsr(3).setRqsGradient(true) 
    def b = new Tsr(-4)
    def w = new Tsr(2)
        
    def y = new Tsr([x, b, w], '((i0+i1)*i2)^2')
    
    /*
     *   f(x) = ((x-4)*2)^2; :=>  f(3) = 4
     *   f(x)' = 8*x - 32 ;  :=>  f(3)' = -8
     *   
     *   y.toString(): "(1):[4.0]; ->d(1):[-8.0]"    
     */
```
Matrix multiplication:
```java
    x = new Tsr(
                [2, 3, 1],
                [
                        3,   2, -1,
                        -2,  2,  4
                ]
    )
    y = new Tsr(
            [1, 3, 2],
            [
                    4, -1,  
                    3,  2,  
                    3, -1
            ])
    def z = new Tsr([x, y], "I[0] x I[1]")
    
    /*
     *   z.toString(): "(2x1x2):[15.0, 2.0, 10.0, 2.0]"    
     */
```
Convolution:
```java
        x = new Tsr(
                [3, 3],
                [
                         1, 2, 5,
                        -1, 4,-2,
                        -2, 3, 4,
                ]
        );
        y = new Tsr(
                [2, 2],
                [
                       -1, 3,
                        2, 3,
                ]);
        z = new Tsr([x, y], 'I[0] x I[1]') 

        // z.toString(): "(2x2):[15.0, 15.0, 18.0, 8.0)]"

        z.backward(new Tsr([2, 2], 1));
        /*
         *   y.toString(): "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```

GPU execution:
```java
        def gpu = Device.find('nvidia')
        x = new Tsr(
                [3, 3],
                [
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                ]
        )
        y = new Tsr(
                [2, 2],
                [
                        -1, 3,
                        2, 3,
                ])
        gpu.store(x).store(y)      
        z = new Tsr([x, y], 'I[0]xI[1]'); // <= executed on gpu!

        // z.toString(): "(2x2):[15.0, 15.0, 18.0, 8.0], "

        z.backward(new Tsr([2, 2], 1))
        /*
         *   y.toString(): "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```
