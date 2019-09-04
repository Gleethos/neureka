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
        //product.backward(new T(product._shape(), 1));
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
        //product.backward(new T(product._shape(), 1));
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

        //int[] expectedAnchor = {1, 4, 4*2, 4*2*9, 4*2*9*5, 4*2*9*5*6, 4*2*9*5*6*2};
        //result =  T.utility.idxTln(dim);
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  T.utility.idxTln(dim);
        printSessionStart("Testing T.utility: dimension _translation!");
        assertEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }
    public int testTensorBase_idxFromAnchor(int[] dim, int idx, int[] expected){
        int [] result =  T.utility.IdxToShpIdx(idx, T.utility.idxTln(dim));
        printSessionStart("Testing T.utility: _shape to _translation to index!");
        assertEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensMulOn(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){

        printSessionStart("Test T.utility: tensMul_mxd");
        int rank = (frstShp.length+scndShp.length)/2;
        int[] frm = new int[rank];
        for(int i=0; i<rank; i++){frm[i]=i;}
        int[][] frstMxd = T.utility.reshapedAndToMxd(frstShp, frm);
        int[][] scndMxd = T.utility.reshapedAndToMxd(scndShp, frm);
        int[][] drnMxd  = T.utility.setupMxdOfCon(frstMxd, scndMxd);
        double[] rsltData = new double[T.utility.szeOfShp(drnMxd[0])];
        T.utility.tensMul_mxd(
            rank,
            new double[][]{frstData, scondData, rsltData},
            new int[]{0, 0, 0},
            frstMxd, scndMxd, drnMxd
        );
        assertEqual(stringified(rsltData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }




    public int testInvTensMulOn(
            int[] frstShp, int[] scndShp,
            double[] frstData, double[] scondData, double[] drnData,
            double[] expctd, boolean first
    ){
        printSessionStart("Test T.utility: tensMul_mxd");
        int rank = (frstShp.length+scndShp.length)/2;
        int[] frm = new int[rank];
        for(int i=0; i<rank; i++){frm[i]=i;}
        int[][] frstMxd = T.utility.reshapedAndToMxd(frstShp, frm);
        int[][] scndMxd = T.utility.reshapedAndToMxd(scndShp, frm);
        int[][] drnMxd  = T.utility.setupMxdOfCon(frstMxd, scndMxd);
        //double[] rsltData = drnData;//new double[T.utility.szeOfShp(drnMxd[0])];
        T.utility.tensMul_inv_mxd(
                rank,
                new double[][]{frstData, scondData, drnData},
                new int[]{0, 0, 0},
                frstMxd, scndMxd, drnMxd,
                first
        );
        assertEqual(stringified((first)?frstData:scondData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }




}
