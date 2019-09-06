package neureka.unit.cases;

import neureka.core.T;

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
            this.assertContains("result", result, element);
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
            this.assertContains("result", result, element);
        }
        for(int i=0; i<source.length; i++){
            this.assertEqual(stringified(source[i].gradient()), stringified(expectedGradient[i]));
        }
        return (printSessionEnd()>0)?1:0;
    }


    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){
        int[] result = T.utility.reshaped(dim, newForm);
        printSessionStart("Testing T.utility: dimension reshaping!");
        assertEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  T.utility.idxTln(dim);
        printSessionStart("Testing T.utility: dimension _translation!");
        assertEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }
    public int testTensorBase_idxFromAnchor(int[] dim, int idx, int[] expected){
        int [] result =  T.utility.idxOf(idx, T.utility.idxTln(dim));
        printSessionStart("Testing T.utility: _shape to _translation to index!");
        assertEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensCon(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){
        printSessionStart("Test T.utility: tensMul_mxd");
        int[] drnMxd  = T.utility.shpOfCon(frstShp, scndShp);
        double[] rsltData = new double[T.utility.szeOfShp(drnMxd)];
        T.utility.tensMul(
                new T(drnMxd, rsltData),
                new T(frstShp, frstData),
                new T(scndShp, scondData)
        );
        assertEqual(stringified(rsltData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }

    public int testInvTensCon(
            int[] frstShp, int[] scndShp,
            double[] frstData, double[] scondData, double[] drnData,
            double[] expctd, boolean first
    ){
        printSessionStart("Test T.utility: tensMul_mxd");
        int[] drnMxd  = T.utility.shpOfCon(frstShp, scndShp);
        T.utility.tensMul_inv(
                new T(frstShp, frstData),
                (first)?new T(scndShp, scondData):new T(drnMxd, drnData),
                (first)?new T(drnMxd, drnData):new T(scndShp, scondData)
        );
        assertEqual(stringified((first)?frstData:scondData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }




}
