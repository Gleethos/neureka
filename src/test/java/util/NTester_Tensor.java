package util;

import neureka.core.T;
import neureka.core.function.IFunction;
import neureka.core.function.factory.Function;
import neureka.core.function.factory.assembly.FunctionGraphBuilder;

public class NTester_Tensor extends NTester {

    public NTester_Tensor(String name)
    {
        super(name);
    }

    public int testTensorAutoGrad(T[] source, String operation, String[] expected){
        printSessionStart("Testing T: autograd!");
        println(bar+"  IFunction: "+operation);
        println(bar+"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        T product = new T(source, operation);
        String result = product.toString("r");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorAutoGrad(T[] source, String operation, String[] expected, T error, double[][] expectedGradient){
        printSessionStart("Testing T: autograd!");
        println(bar+"  IFunction: "+operation);
        println(bar+"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        T product = new T(source, operation);
        product.backward(error);
        String result = product.toString("r");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        for(int i=0; i<source.length; i++){
            this.assertIsEqual(stringified(source[i].gradient()), stringified(expectedGradient[i]));
        }
        return (printSessionEnd()>0)?1:0;
    }


    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){
        int[] result = T.factory.util.rearrange(dim, newForm);
        printSessionStart("Testing T.util: dimension reshaping!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  T.factory.util.idxTln(dim);
        printSessionStart("Testing T.util: dimension _translation!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }
    public int testTensorBase_idxFromAnchor(int[] dim, int idx, int[] expected){
        int [] result =  T.factory.util.idxOf(idx, T.factory.util.idxTln(dim));
        printSessionStart("Testing T.util: _shape to _translation to index!");
        assertIsEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensCon(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test T.util: tensMul_mxd");
        int[] drnMxd  = T.factory.util.shpOfCon(frstShp, scndShp);
        double[] rsltData = new double[T.factory.util.szeOfShp(drnMxd)];
        Function.exec.tensMul(
                new T(drnMxd, rsltData),
                new T(frstShp, frstData),
                new T(scndShp, scondData)
        );
        assertIsEqual(stringified(rsltData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    public int testInvTensCon(
            int[] frstShp, int[] scndShp,
            double[] frstData, double[] scondData, double[] drnData,
            double[] expctd, boolean first
    ){
        printSessionStart("Test T.util: tensMul_mxd");
        int[] drnMxd  = T.factory.util.shpOfCon(frstShp, scndShp);
        Function.exec.tensMul_inv(
                new T(frstShp, frstData),
                (first)?new T(scndShp, scondData):new T(drnMxd, drnData),
                (first)?new T(drnMxd, drnData):new T(scndShp, scondData)
        );
        assertIsEqual(stringified((first)?frstData:scondData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    /**
     * @param tensors
     * @param f
     * @return
     */
    public int testInjection(T[] tensors, String f, String[][] expected)
    {
        printSessionStart("Test injection: I[0] <- I[1], I[0] -> I[1] : "+f);
        T[] result = new T[tensors.length+1];
        result[0] = new T(tensors, f);
        for(int i=1; i<result.length; i++){
            result[i] = tensors[i-1];
        }
        for(int i=0; i<result.length; i++){
            if(expected[i]!=null && expected[i].length!=0){
                String[] parts = expected[i];
                String str = result[i].toString("rdg");
                for(String part : parts){
                    this.assertStringContains("tensor["+i+"]", str, part);
                }
            }
        }
        return (printSessionEnd()>0)?1:0;
    }


}
