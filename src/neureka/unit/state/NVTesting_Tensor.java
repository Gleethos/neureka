package neureka.unit.state;

import neureka.main.core.base.data.NTensor;
import neureka.unit.NVTesting;
import neureka.utility.NMessageFrame;

public class NVTesting_Tensor extends NVTesting {

    public NVTesting_Tensor(NMessageFrame console, NMessageFrame resultConsole)
    {
        super(console, resultConsole);
    }

    public int testTensorAutoGrad(NTensor[] source, String operation, NTensor expected){

        printStart("Testing NTensor.utility: dimension reshaping!");

        NTensor product = new NTensor().of(source, operation);

        product.forward();

        this.assertEqual(product.toString(), expected.toString());



        return 0;
    }

    public int testTensorUtility_reshape(int[] dim, int[] newForm, int[] expected){

        int[] result = NTensor.utility.reshaped(dim, newForm);
        printStart("Testing NTensor.utility: dimension reshaping!");
        if(assertEqual(stringified(result), stringified(expected))){
            return 1;
        }
        int[] expectedAnchor = {1, 4, 4*2, 4*2*9, 4*2*9*5, 4*2*9*5*6, 4*2*9*5*6*2};
        result =  NTensor.utility.idxTrltn(dim);


        return 0;
    }

    public int testTensorUtility_translation(int[] dim, int[] expected){
        int [] result =  NTensor.utility.idxTrltn(dim);
        printStart("Testing NTensor.utility: dimension anchoring!");
        if(assertEqual(stringified(result), stringified(expected))){
            return 1;
        }
        return 0;
    }
    public int testTensorBase_idxFromAnchor(int[] dim, int idx, int[] expected){
        int [] result =  NTensor.utility.IdxToShpIdx(idx, NTensor.utility.idxTrltn(dim));
        printStart("Testing NTensor.utility: dim to anchor to idxs!");
        if(assertEqual(stringified(result), stringified(expected))){
            return 1;
        }
        return 0;
    }

    public int testTensMulOnMxd(){
        int TestCounter = 0;
        int SuccessCounter = 0;
        //---
        int[] frstShp = {2, 3, 4, 3};
        int[] scndShp = {3, 2, 2};

        int[] frstFrm = {0, 1, 2, 3};
        int[] scndFrm = {1, 0, 2, -1};

        int[] drnShp = {1, 1, 3, 3};
        double[] frstData = new double[NTensor.utility.sizeOfShape_mxd(frstShp, 0, frstShp.length)];
        double[] scndData = new double[NTensor.utility.sizeOfShape_mxd(scndShp, 0, scndShp.length)];
        double[] drnData = new double[NTensor.utility.sizeOfShape_mxd(drnShp, 0, drnShp.length)];

        int[][] frstMxd = NTensor.utility.reshapedAndToMxd(frstShp, frstFrm);
        int[][] scndMxd = NTensor.utility.reshapedAndToMxd(scndShp, scndFrm);
        int[][] drnMxd  = NTensor.utility.mxdFromShape(drnShp);

        NTensor.utility.tensMul_mxd(4, new double[][]{frstData, scndData, drnData}, new int[]{0,0,0}, frstMxd, scndMxd, drnMxd);
        int[] xpctdRslt = {3, 2, 1, 2};

        TestCounter++;
        return 1;
    }

    public int testTensMulOn(int[] frstShp, int[] scndShp, double[] frstData, double[] scondData, double[] expctd){

        printStart("Test NTensor.utility: tensMul_mxd");
        int rank = (frstShp.length+scndShp.length)/2;
        int[] frm = new int[rank];
        for(int i=0; i<rank; i++){frm[i]=i;}
        int[][] frstMxd = NTensor.utility.reshapedAndToMxd(frstShp, frm);
        int[][] scndMxd = NTensor.utility.reshapedAndToMxd(scndShp, frm);
        int[][] drnMxd  = NTensor.utility.resultMxdOf(frstMxd, scndMxd);
        double[] rsltData = new double[NTensor.utility.sizeOfShape_mxd(drnMxd[0], drnMxd[3][0], rank)];
        NTensor.utility.tensMul_mxd(
            rank,
            new double[][]{frstData, scondData, rsltData},
            new int[]{0, 0, 0},
            frstMxd, scndMxd, drnMxd
        );
        assertEqual(stringified(rsltData), stringified(expctd));
        return 0;
    }



}
