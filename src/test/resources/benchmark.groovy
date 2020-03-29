
   import neureka.Tsr
   import neureka.acceleration.Device

   import java.nio.file.Files
   import java.nio.file.Paths

   return (Map<String, Integer> conf, String filename, Device device) ->
   {
      // CORE BENCHMARK CODE START:
      def benchmark =  (int iterations, int difficulty) ->
      {
         Map<String, List> map = [:]
         int N, size
         long time
         double delta
         Closure execute = (Closure c) -> c()
         Closure measure = (String attribute_name, Closure c) -> {
            time = System.nanoTime()
            c()
            delta = (System.nanoTime() - time) / 1_000_000_000
            map[attribute_name] = [delta]
         }
         //==========================================================================#
         // Matrix multiplication
         N = 1 * iterations
         size = 1 * difficulty
         //-------------
         execute {
            Tsr A = new Tsr([size, size, 1], "apple").add(device)
            Tsr B = new Tsr([1, size, size], "banana").add(device)
            measure "matrix_multiplication", {
               for (int i; i < N; i++) "I[0]xI[1]" % [A, B]
            }
         }
         //==========================================================================#
         // Vector multiplication
         N = 1 * iterations
         size = 1 * difficulty**2
         //-------------
         execute {
            Tsr C = new Tsr([size], "blueberry").add(device)
            Tsr D = new Tsr([size], "grapefruit").add(device)
            time = System.nanoTime()
            measure "vector_multiplication", {
               for (int i; i < N; i++) "I[0]xI[1]" % [C, D]
            }
         }
         //==========================================================================#
         // Manual Convolution
         N = 1 * iterations
         size = 1 * difficulty
         //-------------
         execute {
            Tsr a = new Tsr([size, size], 3..19).add(device)
            measure "manual_convolution", {
               for (int i; i < N; i++) {
                  Tsr rowconvol = a[1..-2, 0..-1] + a[0..-3, 0..-1] + a[2..-1, 0..-1]//(98, 100) (98, 100) (98, 100)
                  Tsr colconvol = rowconvol[0..-1, 1..-2] + rowconvol[0..-1, 0..-3] + rowconvol[0..-1, 2..-1] - 9 * a[1..-2, 1..-2]
                  //Example for size=100 : (98, 98)+(98, 98)+(98, 98)-9*(98, 98)
               }
            }
         }
         //==========================================================================#
         // Tensor Math
         N = 1 * iterations
         size = 1 * difficulty
         //-------------
         execute {
            def dim = [Math.floor(size / 10), Math.floor(size / 6), Math.floor(size / 3)]
            dim[0] = ((dim[0] == 0) ? 2 : dim[0])
            dim[1] = ((dim[1] == 0) ? 2 : dim[1])
            dim[2] = ((dim[2] == 0) ? 1 : dim[2])
            Tsr t1 = new Tsr(dim).add(device)
            Tsr t2 = new Tsr(dim).add(device)
            measure "tensor_math", {
               for (int i; i < N; i++) {
                  Tsr v = t1 * 10
                  v = v * t2 / t1
                  v = v ** 0.5
               }
            }
         }
         //==========================================================================#
         map["iterations"] = [iterations]
         map["difficulty"] = [difficulty]
         //==========================================================================#

         //Neureka.instance().settings().show()

         return map
      }
      // CORE BENCHMARK CODE END;

      def result_map = [:]
      def result
      BufferedWriter writer = Files.newBufferedWriter(Paths.get("docs/benchmarks/"+filename));
      writer.write("")
      writer.flush()
      for(i in conf["difficulty"]..conf["difficulty"]+conf["sample_size"]){
         result = benchmark(
                 conf["iterations"],
                 i+(i-conf["difficulty"])*conf["intensifier"]
         )
         result.each(k, v)->{
            if (result_map[k]==null) result_map[k] = v
            else result_map[k] += v
         }
      }
      File asCSV = new File("docs/benchmarks/"+filename)
      result_map.each(k, v)->{
         asCSV.append(k)
         asCSV.append(",")
      }
      asCSV.append("\n")
      for(i in 1..conf["sample_size"]){
         result_map.each((k, v)->{
            asCSV.append(v[i-1])
            asCSV.append(",")
         })
         asCSV.append("\n")
      }
   }

