
//   import neureka.Tsr
//
//   def N
//   def size
//   def t
//   //==========================================================================#
//
//   // Matrix multiplication
//   N = 20
//   //-------------
//   size = 128//4096
//   Tsr A = new Tsr([size, size, 1], "hei")
//   Tsr B = new Tsr([1, size, size], "hui")
//   t = System.nanoTime()
//   for(int i : 0..N) new Tsr([A, B],"I[0]xI[1]")
//   delta = System.nanoTime() - t
//   println("Matrix mul")
//   A = null
//   B = null
//
//   //==========================================================================#
//
//   // Vector multiplication
//   N = 100
//   //-------------
//   Tsr C = new Tsr([size * 128])
//   Tsr D = new Tsr([size * 128])
//   t = System.nanoTime()
//   for(int i : 0..N) new Tsr([C, D],"I[0]xI[1]")
//   delta = System.nanoTime() - t
//   println("Dotted two vectors of length "+(size * 128)+" in "+(1e3 * delta / N)+" ms.")
//   C = null
//   D = null
//
//   //==========================================================================#
//   // Manual Convolution
//   N = 1000
//   //-------------
//   Tsr a = new Tsr([100, 100], 3..19)
//   t = System.nanoTime()
//   for(int i : 0..N){
//       Tsr rowconvol = a[1..-2,0..-1] + a[0..-3,0..-1] + a[2..-1,0..-1]//(98, 100) (98, 100) (98, 100)
//       Tsr colconvol = rowconvol[0..-1,1..-2] + rowconvol[0..-1,0..-3] + rowconvol[0..-1,2..-1] - 9*a[1..-2,1..-2]//(98, 98)+(98, 98)+(98, 98)-9*(98, 98)
//   }
//   delta = System.nanoTime() - t
//   println("Convolution of length "+(100*100)+" in "+(1e3 * delta / N)+" ms.")
//   a = null
//
//   //==========================================================================#
//   // Tensor Math
//   N = 100
//   //-------------
//   Tsr t1 = new Tsr([10, 2, 50])
//   Tsr t2 = new Tsr([10, 2, 50])
//   t = System.nanoTime()
//   for(int i : 0..N){
//       Tsr v = t1 * 10
//       v = v * t2 / t1
//       v = v ** 0.5
//   }
//   delta = System.nanoTime() - t
//   println("Convolution of length "+(100*1000)+" in "+(1e3 * delta / N)+" ms.")
//   t1 = null
//   t2 = null
//
//   //==========================================================================#
//
//   print('')
//   print('This was obtained using the following Numpy configuration:')
//
//   np.__config__.show()
