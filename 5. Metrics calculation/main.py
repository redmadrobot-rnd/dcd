import pandas as pd
from tqdm import tqdm
from typing import Dict

from config import Config
from rag_evaluator import RAGMetricsEvaluator


class MetricsCalculator:
    def __init__(self, config: Config):
        self.config = config
        self.rag_evaluator = RAGMetricsEvaluator(config)
        
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_excel(path)
        df.columns = df.columns.str.lower().str.strip()
        
        missing = set(self.config.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        
        return df
    
    def calculate_naive_rag_metrics(self) -> Dict[str, float]:
        print("\n" + "="*60)
        print("NAIVE RAG METRICS (GENERATION + CONTEXT)")
        print("="*60)
        
        df = self.load_dataset(self.config.naive_rag_dataset)
        
        arc_scores, cr_scores, fa_scores, ctx_scores = [], [], [], []
        arc_details, cr_details, fa_details, ctx_details = [], [], [], []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Naive RAG"):
            question = str(row["question"])
            answer = str(row["answer"])
            context = str(row["context"])
            generate_answer = str(row["generate_answer"])
            find_context = str(row["find_context"])
            
            try:
                arc = self.rag_evaluator.evaluate_arc(question, generate_answer)
                arc_pass = int(arc.D and arc.P and arc.Sp and arc.V)
                arc_scores.append(arc_pass)
                arc_details.append(arc.reasoning)
            except Exception as e:
                print(f"\n[ARC error row {idx}] {e}")
                arc_scores.append(0)
                arc_details.append(f"error: {e}")
            
            try:
                cr = self.rag_evaluator.evaluate_cr(question, find_context, generate_answer)
                cr_scores.append(int(cr.verdict))
                cr_details.append(cr.reasoning)
            except Exception as e:
                print(f"\n[CR error row {idx}] {e}")
                cr_scores.append(0)
                cr_details.append(f"error: {e}")
            
            try:
                fa = self.rag_evaluator.evaluate_fa(find_context, generate_answer)
                fa_scores.append(int(fa.verdict))
                fa_details.append(fa.reasoning)
            except Exception as e:
                print(f"\n[FA error row {idx}] {e}")
                fa_scores.append(0)
                fa_details.append(f"error: {e}")
            
            try:
                ctx = self.rag_evaluator.evaluate_context_relevance(question, context, find_context)
                ctx_scores.append(ctx.score)
                ctx_details.append(ctx.explanation)
            except Exception as e:
                print(f"\n[Context error row {idx}] {e}")
                ctx_scores.append(0)
                ctx_details.append(f"error: {e}")
        
        results = {
            "sb_arc": arc_scores, 
            "sb_arc_reasoning": arc_details,
            "sb_cr": cr_scores,
            "sb_cr_reasoning": cr_details,
            "sb_fa": fa_scores,
            "sb_fa_reasoning": fa_details,
            "context_score": ctx_scores,
            "context_explanation": ctx_details
        }
        
        output_path = self.config.results_dir / self.config.detailed_naive_rag
        pd.DataFrame(results).to_excel(output_path, index=False)
        print(f"\nDetailed results saved → {output_path}")
        
        n = len(arc_scores)
        ctx_perfect = sum(1 for s in ctx_scores if s == 2)
        ctx_avg_norm = sum(ctx_scores) / (2 * n)
        
        metrics = {
            "SB_ARC": sum(arc_scores) / n,
            "SB_CR": sum(cr_scores) / n,
            "SB_FA": sum(fa_scores) / n,
            "Context_Perfect": ctx_perfect / n,
            "Context_Avg": sum(ctx_scores) / n,
            "Context_Avg_Norm": ctx_avg_norm
        }
        
        return metrics
    
    def calculate_dcd_metrics(self) -> Dict[str, float]:
        print("\n" + "="*60)
        print("DCD METRICS (GENERATION + CONTEXT)")
        print("="*60)
        
        df = self.load_dataset(self.config.dcd_dataset)
        
        arc_scores, cr_scores, fa_scores, ctx_scores = [], [], [], []
        arc_details, cr_details, fa_details, ctx_details = [], [], [], []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating DCD"):
            question = str(row["question"])
            answer = str(row["answer"])
            context = str(row["context"])
            generate_answer = str(row["generate_answer"])
            find_context = str(row["find_context"])
            
            try:
                arc = self.rag_evaluator.evaluate_arc(question, generate_answer)
                arc_pass = int(arc.D and arc.P and arc.Sp and arc.V)
                arc_scores.append(arc_pass)
                arc_details.append(arc.reasoning)
            except Exception as e:
                print(f"\n[ARC error row {idx}] {e}")
                arc_scores.append(0)
                arc_details.append(f"error: {e}")
            
            try:
                cr = self.rag_evaluator.evaluate_cr(question, find_context, generate_answer)
                cr_scores.append(int(cr.verdict))
                cr_details.append(cr.reasoning)
            except Exception as e:
                print(f"\n[CR error row {idx}] {e}")
                cr_scores.append(0)
                cr_details.append(f"error: {e}")
            
            try:
                fa = self.rag_evaluator.evaluate_fa(find_context, generate_answer)
                fa_scores.append(int(fa.verdict))
                fa_details.append(fa.reasoning)
            except Exception as e:
                print(f"\n[FA error row {idx}] {e}")
                fa_scores.append(0)
                fa_details.append(f"error: {e}")
            
            try:
                ctx = self.rag_evaluator.evaluate_context_relevance(question, context, find_context)
                ctx_scores.append(ctx.score)
                ctx_details.append(ctx.explanation)
            except Exception as e:
                print(f"\n[Context error row {idx}] {e}")
                ctx_scores.append(0)
                ctx_details.append(f"error: {e}")
        
        results = {
            "sb_arc": arc_scores, 
            "sb_arc_reasoning": arc_details,
            "sb_cr": cr_scores,
            "sb_cr_reasoning": cr_details,
            "sb_fa": fa_scores,
            "sb_fa_reasoning": fa_details,
            "context_score": ctx_scores,
            "context_explanation": ctx_details
        }
        
        output_path = self.config.results_dir / self.config.detailed_dcd
        pd.DataFrame(results).to_excel(output_path, index=False)
        print(f"\nDetailed results saved → {output_path}")
        
        n = len(arc_scores)
        ctx_perfect = sum(1 for s in ctx_scores if s == 2)
        ctx_avg_norm = sum(ctx_scores) / (2 * n)
        
        metrics = {
            "SB_ARC": sum(arc_scores) / n,
            "SB_CR": sum(cr_scores) / n,
            "SB_FA": sum(fa_scores) / n,
            "Context_Perfect": ctx_perfect / n,
            "Context_Avg": sum(ctx_scores) / n,
            "Context_Avg_Norm": ctx_avg_norm
        }
        
        return metrics
    
    def run(self):
        print("\n" + "="*60)
        print("STARTING METRICS CALCULATION")
        print("="*60)
        
        naive_rag_metrics = self.calculate_naive_rag_metrics()
        dcd_metrics = self.calculate_dcd_metrics()
        
        self._save_naive_rag_results(naive_rag_metrics)
        self._save_dcd_results(dcd_metrics)
        
        self._print_results(naive_rag_metrics, dcd_metrics)
    
    def _save_naive_rag_results(self, metrics: Dict[str, float]):
        output_path = self.config.results_dir / self.config.naive_rag_metrics_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("NAIVE RAG METRICS\n")
            f.write("="*60 + "\n\n")
            
            f.write("GENERATION METRICS:\n")
            f.write(f"  SB ARC (Answer Relevance & Completeness): {metrics['SB_ARC']:.4f}\n")
            f.write(f"  SB CR  (Context Recall):                  {metrics['SB_CR']:.4f}\n")
            f.write(f"  SB FA  (Factual Accuracy):                {metrics['SB_FA']:.4f}\n")
            
            f.write("\nCONTEXT METRICS:\n")
            f.write(f"  Context Perfect (score == 2):             {metrics['Context_Perfect']:.4f}\n")
            f.write(f"  Context Avg (raw score):                  {metrics['Context_Avg']:.4f}\n")
            f.write(f"  Context Avg Norm (normalized 0-1):        {metrics['Context_Avg_Norm']:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"\nNaive RAG metrics saved → {output_path}")
    
    def _save_dcd_results(self, metrics: Dict[str, float]):
        output_path = self.config.results_dir / self.config.dcd_metrics_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("DCD METRICS\n")
            f.write("="*60 + "\n\n")
            
            f.write("GENERATION METRICS:\n")
            f.write(f"  SB ARC (Answer Relevance & Completeness): {metrics['SB_ARC']:.4f}\n")
            f.write(f"  SB CR  (Context Recall):                  {metrics['SB_CR']:.4f}\n")
            f.write(f"  SB FA  (Factual Accuracy):                {metrics['SB_FA']:.4f}\n")
            
            f.write("\nCONTEXT METRICS:\n")
            f.write(f"  Context Perfect (score == 2):             {metrics['Context_Perfect']:.4f}\n")
            f.write(f"  Context Avg (raw score):                  {metrics['Context_Avg']:.4f}\n")
            f.write(f"  Context Avg Norm (normalized 0-1):        {metrics['Context_Avg_Norm']:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"\nDCD metrics saved → {output_path}")
    
    def _print_results(self, naive_rag_metrics: Dict[str, float], dcd_metrics: Dict[str, float]):
        print("\n" + "="*60)
        print("FINAL METRICS RESULTS")
        print("="*60)
        
        print("\nNaive RAG Metrics:")
        print("-"*60)
        print("GENERATION:")
        print(f"  SB ARC (Answer Relevance & Completeness): {naive_rag_metrics['SB_ARC']:.4f}")
        print(f"  SB CR  (Context Recall):                  {naive_rag_metrics['SB_CR']:.4f}")
        print(f"  SB FA  (Factual Accuracy):                {naive_rag_metrics['SB_FA']:.4f}")
        print("CONTEXT:")
        print(f"  Context Perfect (score == 2):             {naive_rag_metrics['Context_Perfect']:.4f}")
        print(f"  Context Avg (raw):                        {naive_rag_metrics['Context_Avg']:.4f}")
        print(f"  Context Avg Norm (0-1):                   {naive_rag_metrics['Context_Avg_Norm']:.4f}")
        
        print("\nDCD Metrics:")
        print("-"*60)
        print("GENERATION:")
        print(f"  SB ARC (Answer Relevance & Completeness): {dcd_metrics['SB_ARC']:.4f}")
        print(f"  SB CR  (Context Recall):                  {dcd_metrics['SB_CR']:.4f}")
        print(f"  SB FA  (Factual Accuracy):                {dcd_metrics['SB_FA']:.4f}")
        print("CONTEXT:")
        print(f"  Context Perfect (score == 2):             {dcd_metrics['Context_Perfect']:.4f}")
        print(f"  Context Avg (raw):                        {dcd_metrics['Context_Avg']:.4f}")
        print(f"  Context Avg Norm (0-1):                   {dcd_metrics['Context_Avg_Norm']:.4f}")
        print("="*60 + "\n")


def main():
    config = Config()
    calculator = MetricsCalculator(config)
    calculator.run()


if __name__ == "__main__":
    main()
