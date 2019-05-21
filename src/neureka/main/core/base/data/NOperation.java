package neureka.main.core.base.data;

import neureka.main.core.modul.calc.NVFHead;
import neureka.main.core.modul.calc.NVFunction;

import java.util.function.Consumer;

import com.aparapi.Kernel;

public class NOperation {

    NVFunction Function = null;
    NTensor[] Source = null;
    boolean doTensMul = false;

    NOperation(NTensor[] source, int[][] translation, String operation){
        NTensor[] translated = new NTensor[source.length];
        for(int i=0; i<translated.length&&i<translation.length; i++){
            translated[i] = NTensor.factory.reshapedCopyOf(source[i], translation[i]);//source[i].reshaped(translation[i]);
        }
    }
    NOperation(NTensor[] connection, String operation){
        construct(connection, operation);
    }
    NOperation(NTensor connection, String operation){
        construct(new NTensor[]{connection}, operation);
    }

    private void construct(NTensor[] source, String operation){
        Source = source;
        doTensMul = false;//tensmul:
        String replacement = "I[0]";
        if(operation.contains("tensmul")){
            if(operation.contains("tensmul(Ij)")){
                operation = operation.replace("tensmul(Ij)", replacement);
                doTensMul = true;
            }else {
                operation = operation.replace("tensmul", replacement);
                doTensMul = true;
            }
        }
        if(operation.contains("tensDotMul")){
            if(operation.contains("tensDotMul(Ij)")){
                operation = operation.replace("tensDotMul(Ij)", replacement);
                doTensMul = true;
            }else {
                operation = operation.replace("tensDotMul", replacement);
                doTensMul = true;
            }
        }
        if(operation.contains("tm")){
            if(operation.contains("tm(Ij)")){
                operation = operation.replace("tm(Ij)", replacement);
                doTensMul = true;
            }else{
                operation = operation.replace("tm", replacement);
                doTensMul = true;
            }
        }
        if(doTensMul){
            operation = operation.replace("Ij", "I[0]");
            operation = operation.replace("Ii", "I[0]");
            operation = operation.replace("I[i]", "I[0]");
            operation = operation.replace("I[j]", "I[0]");
        }
        Function = new NVFHead();
        Function = Function.newBuild(operation);
        if(!doTensMul && !validSource()){
            this.Function = null;
            this.Source = null;
        }
    }

    private boolean validSource(){
        if(!doTensMul){
            NTensor current = null;
            NTensor last = null;
            int[] shape;
            for(int i=0; i<Source.length; i++){
                current = Source[i];
                if(i>0){
                    if(current.shape().length!=last.shape().length){
                        return false;
                    }
                    for(int j=0; i<current.shape().length; j++){
                        if(current.shape()[j]!=last.shape()[j]){return false;}
                    }
                }
                last = current;
            }
        }
        return true;
    }

    public NTensor tensMul(NTensor tensor1, NTensor tensor2){
        NTensor drn = new NTensor(tensor1.shape());
        int[] index = new int[drn.shape().length];
        int size = drn.size();
        for(int i=0; i<size; i++){
            drn.e_add(index, tensor1.e_get(index)*tensor2.e_get(index));
            NTensor.utility.increment(index, drn.shape());
        }
        return drn;
    }

    public NTensor tensAdd(NTensor tensor1, NTensor tensor2){
        NTensor drn = new NTensor(tensor1.shape());
        int[] index = new int[drn.shape().length];
        int size = drn.size();
        for(int i=0; i<size; i++){
            drn.e_add(index, tensor1.e_get(index)+tensor2.e_get(index));
            NTensor.utility.increment(index, drn.shape());
        }
        return drn;
    }

    public NTensor tensDotMul(NTensor tensor1, NTensor tensor2){
        NTensor newTensor = new NTensor(NTensor.utility.shpOfTensMul(tensor1.shape(), tensor2.shape()));
        NTensor.utility.tensMul_mxd(
                newTensor.shape().length,
                new double[][]{tensor1.value(), tensor2.value(), newTensor.value()}, new int[]{0, 0, 0},
                NTensor.utility.mxdFromShape(tensor1.shape()),
                NTensor.utility.mxdFromShape(tensor2.shape()),
                NTensor.utility.mxdFromShape(newTensor.shape())
        );
        return newTensor;
    }

    public void forward(NTensor tensor){
        if(Source==null || (Function==null && !doTensMul)){
            return;
        }
        boolean carriesDerivatives = (tensor.carriesDerivatives())? this.connectionCarriesDerivatives(): false;
        boolean carriesBackprop = (tensor.carriesBackwardRoutes())?this.connectionCarriesBackpropRoutes():false;
        if(doTensMul){// ((((((A B)C)D)E)G)F)
            NTensor temp =null;
            for(int i = 0; i< Source.length; i++){
                NTensor result = Source[i];
                NTensor second = Source[i];
                NTensor first = temp;
                if(i>0){
                    result = tensDotMul(first, second);
                    //--------------------------------------------------------------------------------------
                    if(carriesDerivatives){
                        NDerivatives d_frst = (NDerivatives) first.findModule(NDerivatives.class);
                        NDerivatives d_scnd = (NDerivatives) second.findModule(NDerivatives.class);
                        if(d_frst!=null||d_scnd!=null){
                            result.addModule(new NDerivatives());
                        }
                        NDerivatives d_rslt = (NDerivatives) result.findModule(NDerivatives.class);
                        if(d_frst!=null && d_scnd==null){
                            d_frst.forEach((NTensor src, NTensor derivative)-> {
                                d_rslt.put(src, tensDotMul(derivative, second));
                            });
                        }
                        if(d_scnd!=null && d_frst==null){
                            d_scnd.forEach((NTensor src, NTensor derivative)-> {
                                d_rslt.put(src, tensDotMul(derivative, first));
                            });
                        }else if(d_scnd!=null && d_frst!=null){
                            d_frst.forEach((src, derivative)-> {
                                d_rslt.put(src, tensDotMul(derivative, second));
                            });
                            d_scnd.forEach((src, derivative)-> {
                                if(d_rslt.has(src)){
                                    d_rslt.get(src).e_add(tensDotMul(derivative, first));
                                }else{
                                    d_rslt.put(src, tensDotMul(derivative, first));
                                }
                            });
                        }
                    }
                    //--------------------------------------------------------------------------------------
                }
                temp = result;
            }
            if(Function!=null){
                this.elementaryActivationOn(new NTensor[]{temp}, tensor, carriesDerivatives);
            }else{
                tensor.copy(temp);
            }
        }else{
            if(Function!=null){
                this.elementaryActivationOn(Source, tensor, carriesDerivatives);
            }
        }
        //carriesDerivatives
        //--------------------------------------------------------------------------------------
        if(tensor.carriesDerivatives()){
            NDerivatives relationalDerivatives = (NDerivatives) tensor.findModule(NDerivatives.class);
            if(tensor.rqsGradient()){
                if(relationalDerivatives==null){
                    relationalDerivatives = new NDerivatives();
                    tensor.addModule(NDerivatives.class);
                }
                relationalDerivatives.put(tensor, new NTensor(tensor.shape(), 1));
            }
        }
        //--------------------------------------------------------------------------------------
        tensor.setSrcDrained(true);
    }

    private void elementaryActivationOn(NTensor[] source, NTensor drain, boolean rqsDerivative){
        if(drain.isEmpty()){
            drain.initialShape(source[0].shape());
        }
        for(int idx = 0; idx<drain.value().length; idx++){
            double[] input = new double[source.length];
            for(int Ii=0; Ii<input.length; Ii++){
                input[Ii] = source[Ii].e_get(source[Ii].shpIdx(idx));
            }
            drain.value()[idx] = Function.activate(input);
            //--------------------------------------------------------------------------------------
            if(rqsDerivative){
                if(!drain.hasModule(NDerivatives.class)){
                    drain.addModule(new NDerivatives());
                }
                NDerivatives drainDeriv = (NDerivatives) drain.findModule(NDerivatives.class);

                double[] d_input = new double[source.length];
                for(int Ii = 0; Ii<source.length; Ii++){
                    NDerivatives relDeriv = (NDerivatives) source[Ii].findModule(NDerivatives.class);
                    if(relDeriv!=null){
                        d_input[Ii] = Function.derive(input, Ii);
                    }
                }
                for(int Ii = 0; Ii< source.length; Ii++){
                    NDerivatives relDeriv = (NDerivatives) source[Ii].findModule(NDerivatives.class);
                    if(relDeriv!=null){
                        if(d_input[Ii]!=0){
                            NDerivatives srcDerivatives = (NDerivatives) source[Ii].findModule(NDerivatives.class);
                            int[] idx_enc = {idx};
                            double[] d_input_enc = {d_input[Ii]};
                            srcDerivatives.forEach(
                                (target, derivative)->{
                                    NTensor found = drainDeriv.get(target);
                                    if(found==null){
                                        found = NTensor.factory.copyOf(derivative);
                                        drainDeriv.put(target, found);
                                    }
                                    found.e_mul(idx_enc[0], d_input_enc[0]);
                                }
                            );
                        }
                    }
                }
            }//DERIVIATION end
            //--------------------------------------------------------------------------------------
        }// Idx loop closed!
    }

    private void exec() {

        Kernel kernel = new Kernel(){
            @Override public void run(){
                int i= getGlobalId();
                //result[i]=intA[i]+inB[i];
            }
        };
        //Range range = Range.create(result.length);
        //kernel.execute(range);

    }


    private boolean connectionCarriesDerivatives(){
        boolean rqsDerivative = false;
        for(int Ii = 0; Ii< Source.length; Ii++){
            if(Source[Ii].hasModule(NDerivatives.class)){
                rqsDerivative = true;
            }
            else if(Source[Ii].rqsGradient()){
                rqsDerivative = true;
            }
        }
        return rqsDerivative;
    }
    private boolean connectionCarriesBackpropRoutes(){
        boolean rqsBackpropRoutes = false;
        for(int Ii = 0; Ii< Source.length; Ii++){
            if(Source[Ii].hasModule(Consumer.class)){
                rqsBackpropRoutes = true;
            }
            else if(Source[Ii].rqsGradient()){
                rqsBackpropRoutes = true;
            }
        }
        return rqsBackpropRoutes;
    }
    /*
     *
     *   tensor1;
     *   tensor2;
     *
     *   tensor3 = new NTensor().of(new NOperation(new NTensor[]{tensor1, tensor2}, "tensDotMul"));
     *   tensor4 = new NOperation(tensor3, "sum(tanh[Ij])").out();
     *
     *
     *
     * */

    public void forEachSource(Consumer<NTensor> action){
        for(int i=0; i<this.Source.length; i++){
            action.accept(Source[i]);
        }
    }


}



