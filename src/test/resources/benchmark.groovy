
   import neureka.Tsr

   return (int iterations, int difficulty)->
   {
      Map<String, List> map = [:]
      int N, size
      long time, delta
      Closure execute = (Closure c)->c()
      //==========================================================================#
      // Matrix multiplication
      N = 1 * iterations
      size = 2 * difficulty
      //-------------
      execute {
         Tsr A = new Tsr([size, size, 1], "banana")
         Tsr B = new Tsr([1, size, size], "apple")
         time = System.nanoTime()
         for (int i; i < N; i++) new Tsr([A, B], "I[0]xI[1]")
         delta = (System.nanoTime() - time)/1_000_000_000
         map["matrix_multiplication"] = [delta]
      }
      //==========================================================================#
      // Vector multiplication
      N = 1 * iterations
      size = 25 * difficulty
      //-------------
      execute {
         Tsr C = new Tsr([size], "blueberry")
         Tsr D = new Tsr([size], "grapefruit")
         time = System.nanoTime()
         for(int i; i<N; i++) new Tsr([C, D],"I[0]xI[1]")
         delta = (System.nanoTime() - time)/1_000_000_000
         map["vector_multiplication"] = [delta]
      }
      //==========================================================================#
      // Manual Convolution
      N = 1 * iterations
      size = 5 * difficulty
      //-------------
      execute {
         Tsr a = new Tsr([size, size], 3..19)
         time = System.nanoTime()
         for(int i; i<N; i++){
            Tsr rowconvol = a[1..-2,0..-1] + a[0..-3,0..-1] + a[2..-1,0..-1]//(98, 100) (98, 100) (98, 100)
            Tsr colconvol = rowconvol[0..-1,1..-2] + rowconvol[0..-1,0..-3] + rowconvol[0..-1,2..-1] - 9*a[1..-2,1..-2]//(98, 98)+(98, 98)+(98, 98)-9*(98, 98)
         }
         delta = (System.nanoTime() - time)/1_000_000_000
         map["manual_convolution"] = [delta]
      }
      //==========================================================================#
      // Tensor Math
      N = 1 * iterations
      size = difficulty
      //-------------
      execute {
         def dim = [Math.floor(size/10), Math.floor(size/6), Math.floor(size/3)]
         dim[0] = ((dim[0]==0)?2:dim[0])
         dim[1] = ((dim[1]==0)?2:dim[1])
         dim[2] = ((dim[2]==0)?1:dim[2])
         Tsr t1 = new Tsr(dim)
         Tsr t2 = new Tsr(dim)
         time = System.nanoTime()
         for(int i; i<N; i++){
            Tsr v = t1 * 10
            v = v * t2 / t1
            v = v ** 0.5
         }
         delta = (System.nanoTime() - time)/1_000_000_000
         map["tensor_math"] = [delta]
      }
      //==========================================================================#
      map["size"] = [size]
      map["difficulty"] = [difficulty]
      //==========================================================================#

      //Neureka.instance().settings().show()

      return map
   }

