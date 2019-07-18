package neureka.unit.state;

import neureka.main.core.base.data.T;
import neureka.main.core.modul.calc.TKernel;
import neureka.unit.NVTesting;
import neureka.utility.NMessageFrame;

public class NVTesting_Tensor extends NVTesting {

    public NVTesting_Tensor(NMessageFrame console, NMessageFrame resultConsole)
    {
        super(console, resultConsole);
    }

    public int testTensorAutoGrad(T[] source, String operation, String expected){
        printSessionStart("Testing T: autograd!");
        println(bar+"  Function: "+operation);
        println(bar+"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        T product = new T().of(source, operation);
        product.backward(new T(product.shape(), 1));
        this.assertEqual(product.toString("r"), expected);
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){

        int[] result = T.utility.reshaped(dim, newForm);
        printSessionStart("Testing T.utility: dimension reshaping!");
        assertEqual(stringified(result), stringified(expected));

        //int[] expectedAnchor = {1, 4, 4*2, 4*2*9, 4*2*9*5, 4*2*9*5*6, 4*2*9*5*6*2};
        //result =  T.utility.idxTrltn(dim);
        return (printSessionEnd()>0)?1:0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  T.utility.idxTrltn(dim);
        printSessionStart("Testing T.utility: dimension translation!");
        assertEqual(stringified(result), stringified(expected));
        return (printSessionEnd()>0)?1:0;
    }
    public int testTensorBase_idxFromAnchor(int[] dim, int idx, int[] expected){
        int [] result =  T.utility.IdxToShpIdx(idx, T.utility.idxTrltn(dim));
        printSessionStart("Testing T.utility: shape to translation to index!");
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
        int[][] drnMxd  = T.utility.resultMxdOf(frstMxd, scndMxd);
        double[] rsltData = new double[T.utility.sizeOfShape_mxd(drnMxd[0], drnMxd[3][0], rank)];
        T.utility.tensMul_mxd(
            rank,
            new double[][]{frstData, scondData, rsltData},
            new int[]{0, 0, 0},
            frstMxd, scndMxd, drnMxd
        );
        assertEqual(stringified(rsltData), stringified(expctd));
        return (printSessionEnd()>0)?1:0;
    }



}
