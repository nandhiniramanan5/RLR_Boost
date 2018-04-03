package edu.wisc.cs.will.ILP.Regression;

import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import edu.wisc.cs.will.Utils.Utils;
import edu.wisc.cs.will.Boosting.Utils.*;


public class BranchStats {
	protected double sumOfOutputSquared = 0;
	protected double sumOfOutput = 0;
	protected double sumOfNumGroundingSquared_pos = 0;
	protected double sumOfNumGroundingSquared_neg = 0;
	protected double sumOfNumGrounding_pos=0;
	protected double sumOfNumGrounding_neg=0;
	protected double sumOfNumGroundingSquaredWithProb = 0;
	protected double sumOfOutputAndNumGrounding_pos = 0;
	protected double sumOfOutputAndNumGrounding_neg = 0;
	protected double sumofNumGrounding_posAndNumGrounding_neg=0;
	protected double numExamples 	=	0;
	private double useFixedLambda = Double.NaN;							 
	protected double numNegativeOutputs = 0;
	protected double numPositiveOutputs = 0;
	protected double w_0 = 0;
	protected double w_1 = 0;
	protected double w_2 = 0;
	
	public void addNumOutput(long num_pos, long num_neg, double output, double weight,double prob) {
		 double deno   = prob * (1-prob);
         if (deno < 0.1) {
         	deno = 0.1; 
         }
//       long num_neg=0;
		sumOfNumGroundingSquared_pos += num_pos*num_pos*weight;
		sumOfNumGroundingSquared_neg += num_neg*num_neg*weight;
		sumOfNumGrounding_pos+= num_pos*weight;	
		sumOfNumGrounding_neg+= num_neg*weight;	
        sumOfOutputAndNumGrounding_pos += num_pos*output*weight;
        sumOfOutputAndNumGrounding_neg += num_neg*output*weight;
        sumofNumGrounding_posAndNumGrounding_neg+=num_pos*weight;
        sumofNumGrounding_posAndNumGrounding_neg+=num_neg*weight;
        sumOfOutputSquared += output * output*weight;
        sumOfOutput+=output*weight;
        
        if (output > 0 ) {
        	numPositiveOutputs+=weight; 
        } else {
        	numNegativeOutputs+=weight;
        }
        numExamples+=weight;
       
        sumOfNumGroundingSquaredWithProb = num_pos*num_pos*weight*deno;        
  
	}
	public BranchStats add(BranchStats other) {
		BranchStats newStats = new BranchStats();
		addTo(other, newStats);
		return newStats;
	}
	
	public double getWeightedVariance(){
		double result = 0;
		double lam=CommandLineArguments.lamda;
		Matrix covariance = DenseMatrix.Factory.zeros(3,3);
			Matrix lamda= DenseMatrix.Factory.zeros(3,3);
			lamda.setAsDouble(lam, 0,0);
			lamda.setAsDouble(lam, 1,1);
			lamda.setAsDouble(lam, 2,2);
			covariance.setAsDouble(1, 0,0);
			covariance.setAsDouble(sumOfNumGrounding_pos, 0,1);
			covariance.setAsDouble(sumOfNumGrounding_neg, 0,2);
			covariance.setAsDouble(sumOfNumGrounding_pos, 1,0);
			covariance.setAsDouble(sumOfNumGroundingSquared_pos, 1,1);
			covariance.setAsDouble(sumofNumGrounding_posAndNumGrounding_neg, 1,2);
			covariance.setAsDouble(sumOfNumGrounding_neg, 2,0);
			covariance.setAsDouble(sumofNumGrounding_posAndNumGrounding_neg, 2,1);
			covariance.setAsDouble(sumOfNumGroundingSquared_neg, 2,2);
			Matrix gradients = DenseMatrix.Factory.zeros(3,1);
			gradients.setAsDouble(sumOfOutput, 0,0);
			gradients.setAsDouble(sumOfOutputAndNumGrounding_pos, 1,0);
			gradients.setAsDouble(sumOfOutputAndNumGrounding_neg, 2,0);
			Matrix weights = DenseMatrix.Factory.zeros(3,1);
			Matrix CT=covariance.transpose();
			Matrix Prod = CT.mtimes(covariance);
//			System.out.println("hi1:: "+Prod);
			Matrix reg = Prod.plus(lamda);
//			System.out.println("hi2:: "+reg);
			Matrix inv = reg.pinv();
			Matrix temp = inv.mtimes(CT);
			Matrix finalm= temp.mtimes(gradients);
			double t1,t2,t3;
			w_0=finalm.getAsDouble(0,0);
			w_1=finalm.getAsDouble(1,0);
			w_2=finalm.getAsDouble(2,0);
			result= (Math.pow((-sumOfOutput+w_0+w_1*sumOfNumGrounding_pos+w_2*sumOfNumGrounding_neg), 2)+(lam/2)*((w_0*w_0)+(w_1*w_1)+(w_2*w_2)));
			return result;
}

	public void addTo(BranchStats other, BranchStats newStats) {
																				  
		newStats.sumOfNumGroundingSquared_pos = this.sumOfNumGroundingSquared_pos + other.sumOfNumGroundingSquared_pos;
		newStats.sumOfNumGroundingSquared_neg = this.sumOfNumGroundingSquared_neg + other.sumOfNumGroundingSquared_neg;
		newStats.sumOfNumGrounding_pos=this.sumOfNumGrounding_pos + other.sumOfNumGrounding_pos;
		newStats.sumOfNumGrounding_neg=this.sumOfNumGrounding_neg + other.sumOfNumGrounding_neg;
		newStats.sumofNumGrounding_posAndNumGrounding_neg=this.sumofNumGrounding_posAndNumGrounding_neg+other.sumofNumGrounding_posAndNumGrounding_neg;
		newStats.sumOfOutputAndNumGrounding_pos = this.sumOfOutputAndNumGrounding_pos + other.sumOfOutputAndNumGrounding_pos;
		newStats.sumOfOutputAndNumGrounding_neg = this.sumOfOutputAndNumGrounding_neg + other.sumOfOutputAndNumGrounding_neg;
		newStats.sumOfOutputSquared = this.sumOfOutputSquared + other.sumOfOutputSquared;
		newStats.sumOfOutput= this.sumOfOutput + other.sumOfOutput;
		newStats.numNegativeOutputs = this.numNegativeOutputs + other.numNegativeOutputs;
		newStats.numPositiveOutputs = this.numPositiveOutputs + other.numPositiveOutputs;
		newStats.numExamples = this.numExamples + other.numExamples;
		newStats.sumOfNumGroundingSquaredWithProb = this.sumOfNumGroundingSquaredWithProb + other.sumOfNumGroundingSquaredWithProb;
		if (!Double.isNaN(this.useFixedLambda) || !Double.isNaN(other.useFixedLambda)) {
			if (this.useFixedLambda != other.useFixedLambda) {
				Utils.waitHere("Different lambdas for " + this.useFixedLambda + " & " + other.useFixedLambda);
			}	else {
				newStats.useFixedLambda = this.useFixedLambda;
			}
		}
	}
	public double getLambda() {
		return getLambda(false);
	}
	public double getLambda2() {
		return getLambda2(false);
	}
	public double getLambda0() {
		return getLambda0(false);
	}
	public double getLambda0(boolean useProbWeights)
	{
		if (sumOfNumGroundingSquared_pos == 0) {
			return 0;
		}
		if (sumOfNumGroundingSquaredWithProb == 0) {
			Utils.waitHere("Groundings squared with prob is 0??");
		}
		double lambda3 =  w_0;		
		return lambda3;
	}
	
	public double getLambda2(boolean useProbWeights)
	{
		if (sumOfNumGroundingSquared_pos == 0) {
			return 0;
		}
		if (sumOfNumGroundingSquaredWithProb == 0) {
			Utils.waitHere("Groundings squared with prob is 0??");
		}
		double lambda2 =  w_2;		
		//if (lambda == 0) {
		//	Utils.println(this.toAttrString());
		//}
		return lambda2;
	}
	public double getLambda(boolean useProbWeights) {
	
		if (!Double.isNaN(useFixedLambda)) {
			return useFixedLambda;
		}
		if (sumOfNumGroundingSquared_pos == 0) {
			return 0;
		}
		if (sumOfNumGroundingSquaredWithProb == 0) {
			Utils.waitHere("Groundings squared with prob is 0??");
		}
		double lambda =  w_1;
		if (useProbWeights) {
			//Utils.waitHere("Computations not correct for vector-based probabilities");
			lambda = sumOfOutputAndNumGrounding_pos / sumOfNumGroundingSquaredWithProb;
		}
		
		//if (lambda == 0) {
		//	Utils.println(this.toAttrString());
		//}
		return lambda;
	}
	

	//MLN Squared Error Calculations

   
																																											 
  
	
	public String toAttrString() {
		return 	"% Sum of Output squared		=	" + sumOfOutputSquared + "\n" +
		//"% Sum of Output 				=	" + sumOfOutput + "\n" +
		"% Sum of # pos groundings squared	=	" + sumOfNumGroundingSquared_pos + "\n" +
		"% Sum of #neg groundings squared	=	" + sumOfNumGroundingSquared_neg + "\n" +
		"% Sum of #groundings^2*Probs	=	" + sumOfNumGroundingSquaredWithProb + "\n" +
		//"% Sum of #groundings 			=	" + sumOfNumGrounding + "\n" +
		"% Sum of #pos groundings*output	=	" + sumOfOutputAndNumGrounding_pos + "\n" +
		"% Sum of #neg groundings*output	=	" + sumOfOutputAndNumGrounding_neg + "\n" +
		"% Sum of #neg *#pos groundings	=	" + sumofNumGrounding_posAndNumGrounding_neg + "\n" +
		"% Num of +ve output			=	" + numPositiveOutputs + "\n" +
		"% Num of -ve output			=	" + numNegativeOutputs ;
	}
	public String toString() {
		return toAttrString() + "\n" + 
				(!Double.isNaN(useFixedLambda) ?
				"% Fixed Lambda					=	" + useFixedLambda + "\n":"") +
				"% Lambda						=	" + getLambda()+ "\n" + 
				"% Lambda2						=	" + getLambda2()+ "\n" + 
				"% Lambda3						=	" + getLambda0()+ "\n" + 
				"% Prob Lambda					=	" + getLambda(true) ;
	}
	
	public void setZeroLambda() {
		useFixedLambda = 0;
	}
	/**
	 * @return the sumOfOutputSquared
	 */
	public double getsumofNumGrounding_posAndNumGrounding_neg() {
		return sumofNumGrounding_posAndNumGrounding_neg;
	}
	
	public double getSumOfOutputSquared() {
		return sumOfOutputSquared;
	}
	/**
	 * @return the sumOfNumGroundingSquared
	 */
	public double getsumOfNumGroundingSquared_pos() {
		return sumOfNumGroundingSquared_pos; 
	}
	public double getsumOfNumGroundingSquared_neg() {
		return sumOfNumGroundingSquared_neg; 
	}
	/**
	 * @return the sumOfNumGroundingSquaredWithProb
	 */
	public double getSumOfNumGroundingSquaredWithProb() {
		return sumOfNumGroundingSquaredWithProb;
	}
	/**
	 * @return the sumOfOutputAndNumGrounding
	 */
	public double getsumOfOutputAndNumGrounding_pos() {
		return sumOfOutputAndNumGrounding_pos;
	}
	public double getsumOfOutputAndNumGrounding_neg() {
		return sumOfOutputAndNumGrounding_neg;
	}
	/**
	 * @return the numExamples
	 */
	public double getNumExamples() {
		return numExamples;
	}
	public double getW0(){
		return w_0;
	}
	public double getW1(){
		return w_1;
	}
	public double getW2(){
		return w_2;
	}
	
	public void setW0(double w0){
		w_0=w0;
	}
	public void setW1(double w1){
		w_1=w1;
	}
	public void setW2(double w2){
		w_2=w2;
	}
	/**
	 * @return the useFixedLambda
	 */
	public double getUseFixedLambda() {
		return useFixedLambda;
	}
	/**
	 * @return the numNegativeOutputs
	 */
	public double getNumNegativeOutputs() {
		return numNegativeOutputs;
	}
	/**
	 * @return the numPositiveOutputs
	 */
	public double getsumOfOutput() {
		return sumOfOutput;
	}
	public double getsumOfNumGrounding_pos() {
		return sumOfNumGrounding_pos;
	}
	public double getsumOfNumGrounding_neg() {
		return sumOfNumGrounding_neg;
	}
	public double getNumPositiveOutputs() {
		return numPositiveOutputs;
	}
	
	
}

