package com.company;

import java.io.*;
import java.util.*;

class NN implements Serializable {

    private Vector<Integer> tplgy;//neuron numbers in each layer  ,  example : 2 4 1 , 12 16 16 8 ...(inputLayer [ hidden Layers ] outputLayer)
    private List<double[][]> weights;//weights between each layer
    private List<double[]> inputs;//input read from data.txt file
    private List<double[]> targets;//read from data.txt
    private List<double[]> outputs;//for using while back prop , it keeps the outputs of each neurons in each layer while feedforwarding
    private List<double[]> gradients;//calculated gradients for each neuron
    public String dataFilePath;

    public void init(){
        //initializing weights

        weights = new ArrayList<>(tplgy.size()-1);//layer number -  1

        gradients = new ArrayList<>(tplgy.size()); //same as layer number ,

        outputs = new ArrayList<>(tplgy.size());//same as layer number
        for(int i=0;i<tplgy.size()-1;i++)
        {
            weights.add(new double[tplgy.get(i)+1][tplgy.get(i+1)]);
            weights.set(i,randomize(weights.get(i)));
        }


        //initializing inputs and targets
        inputs = new ArrayList<>(1000);
        targets = new ArrayList<>(1000);
        try {
            Scanner ms = new Scanner(new File(this.dataFilePath));

            int i = 0;
            while(ms.hasNextLine())
            {
                String line = ms.nextLine();
                String[] columns = line.split(" ");

                double[] tmp = new double[tplgy.get(0)];

                //System.out.print("inputs : ");
                for (int k=0; k<columns.length; k++) {
                    //System.out.println(columns[i]);
                    Double temp = new Double(Double.parseDouble(columns[k]));
                    tmp[k] = temp;
                    //System.out.print(" "+ temp +" ");
                }
                //System.out.println();

                inputs.add(tmp);


                String line2 = ms.nextLine();
                String[] columns2 = line2.split(" ");

                double[] tmp2 = new double[tplgy.lastElement()];

                //System.out.print("targets : ");
                for (int k=0; k<tplgy.lastElement(); k++) {
                    //System.out.println(columns[i]);
                    Double temp = new Double(Double.parseDouble(columns2[k]));
                    tmp2[k] = temp;
                    //System.out.print(" "+ temp +" ");
                }

                System.out.println(i++);

                targets.add(tmp2);
                i++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }



    }

    ///adding bias neuron to each layer while feedforward
    private double[] addBias(final double[] tmp) {

        double [] rtn = new double[tmp.length+1];
        for(int i=0;i<tmp.length;i++)
            rtn[i] = tmp[i];
        rtn[rtn.length-1] = 1.0;

        return rtn;

    }

    int EPOCH = 10000;
    public void train(){
        assert (inputs.size() == targets.size());
        for(int i=0;i<EPOCH;i++)
        {


            for(int j=0;j<inputs.size();j++)
            {
                double[] result = forwardFeed(inputs.get(j));

                reportError(result,targets.get(j));
                backProp(result,targets.get(j));

                //after a feed forward and a backprop , we have nothing to do with current neuron outputs and gradients
                //we have to clear them , or else it does not work
                outputs.clear();
                gradients.clear();
            }

            System.out.println("net AVG ERROR : "+ netRecentAvrgError);


        }

    }

    private void backProp(final double[] result,final double[] ts) {

        double [] deltas = new double[result.length];
        List<double[]> tmp = new ArrayList<>();
        tmp.add(new double[outputs.get(outputs.size()-1).length]);
        for(int i=0;i<tplgy.lastElement();i++)
        {
            deltas[i] = (result[i]-ts[i])*(outputs.get(outputs.size()-1)[i]*(1-outputs.get(outputs.size()-1)[i]));
            tmp.get(0)[i] = deltas[i];
        }

        int index = 0;
        //let s calculate all the neuron gradients through layers
        //
        for(int i=tplgy.size()-2;i>=0;i--)
        {

            double[] darr = new double[outputs.get(i).length];
            for(int j=0;j<outputs.get(i).length;j++)
            {
                    for(int l=0;l<weights.get(i)[0].length;l++)
                    {
                        darr[j] += weights.get(i)[j][l] * tmp.get(index)[l];
                    }
                    darr[j] *= outputs.get(i)[j] * (1-outputs.get(i)[j]);//derivative of sigmoid

            }
            index++;
            tmp.add(darr);
        }

        //
        for(int i=tmp.size()-1;i>=0;i--)
        {
            gradients.add(tmp.get(i));
        }

        // update weights

        for(int i=0;i<weights.size();i++)
        {
            double [] biased = addBias(outputs.get(i));
            for(int j=0;j<weights.get(i).length;j++)
            {
                for(int k=0;k<weights.get(i)[j].length;k++)
                {

                    weights.get(i)[j][k] -= gradients.get(i+1)[k] * biased[j]*0.15;//0.15 is the learning rate
                }
            }
        }




    }

    private double netRecentAvrgError = 0.0;

    private void reportError(final double[] result, final double[] ts) {


        double error = 0.0;

        for(int i=0;i<tplgy.lastElement();i++)
        {
            error += (result[i] - ts[i]) * (result[i] - ts[i]);
        }

        error /= result.length; // get average error squared
        error =Math.sqrt(error); // RMS

        // Implement a recent average measurement

        //the number 1000 is the smoothing factor here
        netRecentAvrgError =(netRecentAvrgError * 1000 + error)/(1000 + 1.0);

        //System.out.println("Error ::: "+netRecentAvrgError);


    }

    public double[] forwardFeed(final double[] input) {

        double[] result = new double[input.length];
        for(int i=0;i<result.length;i++)
            result[i] = input[i];

        outputs.add(result);
        for(int i=0;i<weights.size();i++)
        {
            outputs.add(i+1,mMultiply(addBias(outputs.get(i)),weights.get(i)));
        }



        return outputs.get(outputs.size()-1);

    }

    private double[] mMultiply(final double [] vector,final double[][] ws){

        double [] rtn = new double[ws[0].length];

        for(int i=0;i<ws[0].length;i++)
        {
            double val = 0.0;
            for(int j=0;j<vector.length;j++)
            {
                val += vector[j] * ws[j][i];
            }

            rtn[i] = transferFunc(val);
        }

        return rtn;
    }

    private double transferFunc(double x) {

        return (1/(1+ Math.pow(Math.E, -x)));
    }


    public double[][] randomize(double [][] arg) {
        Random r = new Random();
        for (int i = 0; i < arg.length; i++)
        {
            for (int j = 0; j < arg[i].length; j++) {
                arg[i][j] = r.nextInt() % 2 == 0 ? r.nextDouble() : -1 * r.nextDouble();

                //arg[i][j] = r.nextDouble();
            }

        }
        return arg;

    }

    public NN(Vector<Integer> topology,String dataPath){
        tplgy = topology;this.dataFilePath = dataPath;
    }
}
public class Main {

    public static void main(String[] args) {

        Vector<Integer> topology = new Vector<>();
        topology.add(784);
        topology.add(40);
        topology.add(40);
        topology.add(10);
        NN nn = new NN(topology,"./src/data.txt");
        nn.init();
        nn.train();


    }

}
