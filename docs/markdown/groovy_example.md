# Neureka with Groovy #

Simple scalar calculation:
```groovy
    def x = Tensor.of(3).setRqsGradient(true) 
    def b = Tensor.of(-4)
    def w = Tensor.of(2)
        
    def y = ((x + b) * w)**2
    
    /*
     *   f(x) = ((x-4)*2)**2; :=>  f(3) = 4
     *   f(x)' = 8*x - 32 ;   :=>  f(3)' = -8
     *   
     *   y.toString(): "(1):[4.0]; ->d(1):[-8.0]"    
     */
```
Matrix multiplication:
```groovy
    def x = Tensor.of(
                [2, 3],
                [
                        3,   2, -1,
                        -2,  2,  4
                ]
            )
            
    def y = Tensor.of(
                [3, 2],
                [
                        4, -1,  
                        3,  2,  
                        3, -1
                ]
             )
            
    def z = x.matMul(y)
    
    /*
     *   z.toString(): "(2x2):[15.0, 2.0, 10.0, 2.0]"    
     */
```
Convolution:
```groovy
        x = Tensor.of(
                [3, 3],
                [
                         1, 2, 5,
                        -1, 4,-2,
                        -2, 3, 4,
                ]
        );
        y = Tensor.of(
                [2, 2],
                [
                       -1, 3,
                        2, 3,
                ]);
                
        z = Tensor.of('i0 x i1', x, y) 

        // z.toString(): "(2x2):[15.0, 15.0, 18.0, 8.0)]"

        z.backward(Tensor.of([2, 2], 1));
        /*
         *   y.toString(): "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```

GPU execution:
```groovy
        def gpu = Device.find('nvidia')
        x = Tensor.of(
                [3, 3],
                [
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                ]
        )
        y = Tensor.of(
                [2, 2],
                [
                        -1, 3,
                        2, 3,
                ])
                
        gpu.store(x).store(y)      
        z = Tensor.of('i0 x i1', x, y); // <= executed on gpu!

        // z.toString(): "(2x2):[15.0, 15.0, 18.0, 8.0], "

        z.backward(Tensor.of([2, 2], 1))
        /*
         *   y.toString(): "(2x2):[-1.0, 3.0, 2.0, 3.0]:g:[6.0, 9.0, 4.0, 9.0]"    
         */
```
