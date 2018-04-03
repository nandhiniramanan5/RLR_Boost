/**
 * 
 */
package edu.wisc.cs.will.ILP.Regression;

import java.util.List;

import edu.wisc.cs.will.Boosting.Common.RunBoostedModels;
import edu.wisc.cs.will.Boosting.MLN.RunBoostedRLR;
import edu.wisc.cs.will.Boosting.RDN.RegressionRDNExample;
import edu.wisc.cs.will.DataSetUtils.Example;
import edu.wisc.cs.will.DataSetUtils.RegressionExample;
import edu.wisc.cs.will.FOPC.Literal;
import edu.wisc.cs.will.ILP.LearnOneClause;
import edu.wisc.cs.will.ILP.SingleClauseNode;
import edu.wisc.cs.will.Utils.ProbDistribution;
import edu.wisc.cs.will.Utils.Utils;
import edu.wisc.cs.will.stdAIsearch.SearchInterrupted;

/**
 * @author tkhot
 *
 */
public class RegressionInfoHolderForRDN extends RegressionInfoHolder {
	double totalCount;
	public RegressionInfoHolderForRDN() {
		trueStats = new BranchStats();
		falseStats = new BranchStats();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#weightedVarianceAtSuccess()
	 */
	@Override
	public double weightedVarianceAtSuccess() {		
		return trueStats.getWeightedVariance();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#weightedVarianceAtFailure()
	 */
	@Override
	public double weightedVarianceAtFailure() {
		return falseStats.getWeightedVariance();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#totalExampleWeightAtSuccess()
	 */
	@Override
	public double totalExampleWeightAtSuccess() {
		return trueStats.getNumExamples();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#totalExampleWeightAtFailure()
	 */
	@Override
	public double totalExampleWeightAtFailure() {
		return falseStats.getNumExamples();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#meanAtSuccess()
	 */
	@Override
	public double meanAtSuccess() {
		return trueStats.getLambda();
	}
	@Override
	public double meanAtSuccess2() {
		return trueStats.getLambda2();
	}
	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#meanAtSuccess()
	 */
	@Override
	public double meanAtSuccess0() {
		return trueStats.getLambda0();
	}
	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#meanAtFailure()
	 */
	@Override
	public double meanAtFailure() {
		return falseStats.getLambda();
	}

	/* (non-Javadoc)
	 * @see edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder#addFailureStats(edu.wisc.cs.will.ILP.Regression.RegressionInfoHolder)
	 */
	@Override
	public RegressionInfoHolder addFailureStats(RegressionInfoHolder addThis) {
		RegressionInfoHolderForRDN regHolder = new RegressionInfoHolderForRDN();
		if (addThis == null) {
			regHolder.falseStats = this.falseStats.add(new BranchStats());
		} else {
			regHolder.falseStats = this.falseStats.add(((RegressionInfoHolderForRDN)addThis).falseStats);
		}
		return regHolder;
	}


	@Override
	public void addFailureExample(Example eg, long numGrndg, long neg, double weight) {
		double output =  ((RegressionExample) eg).getOutputValue();
		ProbDistribution prob   = ((RegressionRDNExample)eg).getProbOfExample();
		if (prob.isHasDistribution()) {
			Utils.error("Expected single probability value but contains distribution");
		}
		falseStats.addNumOutput(numGrndg, neg, output, weight, prob.getProbOfBeingTrue());
	}

	@Override
	public double variance() {
		return (weightedVarianceAtSuccess() + weightedVarianceAtFailure()) / (totalExampleWeightAtSuccess() + totalExampleWeightAtFailure());
	}

	@Override
	public void populateExamples(LearnOneClause task, SingleClauseNode caller) throws SearchInterrupted {
		if (!task.regressionTask) { Utils.error("Should call this when NOT doing regression."); }
		if (caller.getPosCoverage() < 0.0) { caller.computeCoverage(); }
		for (Example posEx : task.getPosExamples()) {
			double weight = posEx.getWeightOnExample();
			double output = ((RegressionExample) posEx).getOutputValue();
			ProbDistribution prob   = ((RegressionRDNExample)posEx).getProbOfExample();
			if (prob.isHasDistribution()) {
				Utils.error("Expected single probability value but contains distribution");
			}
			if (!caller.posExampleAlreadyExcluded(posEx)) {
				if(RunBoostedModels.cmdGlob.isLearnMLN())
				{
				int clSize = caller.getClause().getDefiniteClauseBody().size();
				String bitrep = "";
				for(int bitlen=0;bitlen<clSize;bitlen++)
					bitrep += "1";
				
				//Utils.waitHere("args in exampleYYYYYYYYYYYYYYYYYYYYYYYYYYYY______________YYYYYYYY"+cl.getDefiniteClauseBody());
				List<Literal> preds = caller.getClause().getDefiniteClauseBody();
//				System.out.println(caller.getClause().getDefiniteClauseBody()+"---------"+RunBoostedRLR.gdb);
				
				
				
				totalCount = RunBoostedRLR.gdb.totalCount(caller.getClause().getDefiniteClauseHead().getArguments().toString().replace("[", "").replace("]", "").replace(", ", ","),
						posEx.asLiteral().getArguments().toString().replace("[", "").replace("]", "").replace(", ", ","), preds, bitrep);
				}
				Double total = totalCount;
				long  neg = (long) (total-1);
				if (total==0)
				{
					neg=0;
				}
				trueStats.addNumOutput(1, neg, output, weight, prob.getProbOfBeingTrue());		
			}
		}
		RegressionInfoHolder totalFalseStats = caller.getTotalFalseBranchHolder() ;
		if (totalFalseStats != null) {
			falseStats = falseStats.add(((RegressionInfoHolderForRDN)totalFalseStats).falseStats);
		}
		// Utils.println("Populated examples: " + trueStats.getNumExamples() + " task: " + caller.getClause());
	}

}
