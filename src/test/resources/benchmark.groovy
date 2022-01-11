
   import neureka.Tsr
   import neureka.devices.Device

   import java.nio.file.Files
   import java.nio.file.Paths

   return (Map<String, Object> conf, String filename, Device device, Closure tester) ->
   {
      // CORE BENCHMARK CODE START:
      def benchmark =  (int iterations, int difficulty) ->
      {
         Map<String, List> map = [:]
         long time
         double delta
         Closure measure = (String attribute_name, Closure c) -> {
            time = System.nanoTime()
            c()
            delta = (System.nanoTime() - time) / 1_000_000_000
            map[attribute_name] = [delta]
         }
         if ( conf.containsKey('custom_code') ) { // A benchmark for custom code is being performed!
            conf['custom_code'].each( pair -> measure( pair.key, {pair.value(iterations, difficulty)} ) )
            map["iterations"] = [iterations]
            map["difficulty"] = [difficulty]
            return map
         }
         Closure execute = (Closure c) -> c()
         int N, size

         //==========================================================================#
         // Matrix multiplication
         N = 1 * iterations
         size = 1 * difficulty
         //-------------
         execute {
            Tsr A = Tsr.of([size, size, 1], "apple").to(device)
            Tsr B = Tsr.of([1, size, size], "banana").to(device)
            measure "matrix_multiplication", {
               for (int i=0; i < N; i++) tester("I[0]xI[1]" % [A, B])
            }
         }
         //==========================================================================#
         // Vector multiplication
         N = 1 * iterations
         size = 1 * difficulty**2
         //-------------
         execute {
            Tsr C = Tsr.of([size], "blueberry").to(device)
            Tsr D = Tsr.of([size], "grapefruit").to(device)
            time = System.nanoTime()
            measure "vector_multiplication", {
               for ( int i=0; i < N; i++) tester("I[0]xI[1]" % [C, D])
            }
         }
         //==========================================================================#
         // Manual Convolution
         N = 1 * iterations
         size = 1 * difficulty
         //-------------
         execute {
            Tsr a = Tsr.of([size, size], 3d..19d).to(device)
            measure "manual_convolution", {
               for ( int i; i < N; i++ ) {
                  Tsr rowconvol = a[1..-2, 0..-1] + a[0..-3, 0..-1] + a[2..-1, 0..-1]//(98, 100) (98, 100) (98, 100)
                  Tsr colconvol = rowconvol[0..-1, 1..-2] + rowconvol[0..-1, 0..-3] + rowconvol[0..-1, 2..-1] - 9 * a[1..-2, 1..-2]
                  tester(colconvol)
                  // Example for size = 100 : (98, 98)+(98, 98)+(98, 98)-9*(98, 98)
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
            Tsr t1 = Tsr.ofShape(dim).to(device)
            Tsr t2 = Tsr.ofShape(dim).to(device)
            measure "tensor_math", {
               for ( int i; i < N; i++ ) {
                  Tsr v = t1 * 10
                  v = v * t2 / t1
                  v = v ** 0.5
                  tester(v)
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
      for(i in conf["difficulty"]..conf["difficulty"]+conf["sample_size"]){
         result = benchmark(
                 conf["iterations"],
                 conf["difficulty"]+(i-conf["difficulty"])*conf["intensifier"]
         )
         result.each(k, v)->{
            if (result_map[k]==null) result_map[k] = v
            else result_map[k] += v
         }
      }
      if(filename!=null){
         BufferedWriter writer = Files.newBufferedWriter(Paths.get("docs/benchmarks/"+filename));
         writer.write("")
         writer.flush()
         File asCSV = new File("docs/benchmarks/"+filename)
         def ci = 0
         def rowSize = result_map.size()
         result_map.each(k, v)->{
            asCSV.append(k)
            if(ci<rowSize-1) asCSV.append(",")
            ci++
         }
         asCSV.append("\n")
         for(i in 1..conf["sample_size"]){
            ci = 0
            result_map.each((k, v)->{
               asCSV.append(v[i-1])
               if(ci<rowSize-1) asCSV.append(",")
               ci++
            })
            asCSV.append("\n")
         }
      }
   }

