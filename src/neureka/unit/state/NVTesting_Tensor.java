package neureka.unit.state;

import neureka.main.core.base.data.T;
import neureka.unit.NVTesting;
import neureka.utility.NMessageFrame;

public class NVTesting_Tensor extends NVTesting {

    public NVTesting_Tensor(NMessageFrame console, NMessageFrame resultConsole)
    {
        super(console, resultConsole);
    }

    public int testTensorAutoGrad(T[] source, String operation, T expected){

        printStart("Testing T.utility: dimension reshaping!");
        T product = new T().of(source, operation);
        product.backward(new T(product.shape(), 1));

        if(this.assertEqual(product.toString(), expected.toString())){
            return 1;
        }
        return 0;
    }

    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){

        int[] result = T.utility.reshaped(dim, newForm);
        printStart("Testing T.utility: dimension reshaping!");
        if(assertEqual(stringified(result), stringified(expected))){
            return 1;
        }
        //int[] expectedAnchor = {1, 4, 4*2, 4*2*9, 4*2*9*5, 4*2*9*5*6, 4*2*9*5*6*2};
        //result =  T.utility.idxTrltn(dim);
        return 0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  T.utility.idxTrltn(dim);
        printStart("Testing T.utility: dimension anchoring!");
        if(assertEqual(stringified(result), stringified(expected))){
            return 1;
        }
        return 0;
    }
    public int testTensorBase_idxFromAnchor(int[] dim, int idx, int[] expected){
        int [] result =  T.utility.IdxToShpIdx(idx, T.utility.idxTrltn(dim));
        printStart("Testing T.utility: dim to anchor to idxs!");
        if(assertEqual(stringified(result), stringified(expected))){
            return 1;
        }
        return 0;
    }

    public int testTensMulOn(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){

        printStart("Test T.utility: tensMul_mxd");
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
        return 0;
    }



}
