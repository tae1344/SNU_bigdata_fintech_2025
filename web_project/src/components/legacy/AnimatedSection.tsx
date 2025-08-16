"use client";
import React from "react";
import { motion, AnimatePresence, Variants } from "motion/react";

interface AnimatedSectionProps {
  children: React.ReactNode;
  isVisible: boolean;
  direction?: "left" | "right" | "up" | "down";
  className?: string;
}

export function AnimatedSection({ 
  children, 
  isVisible, 
  direction = "right",
  className = "" 
}: AnimatedSectionProps) {
  const getVariants = (): Variants => {
    const baseVariants: Variants = {
      hidden: { opacity: 0 },
      visible: { 
        opacity: 1,
        transition: {
          duration: 0.6,
          ease: "easeOut", // 올바른 ease 타입 사용
          staggerChildren: 0.1
        }
      },
      exit: { 
        opacity: 0,
        transition: {
          duration: 0.4,
          ease: "easeIn" // 올바른 ease 타입 사용
        }
      }
    };

    switch (direction) {
      case "left":
        return {
          hidden: { 
            opacity: 0,
            x: 100,
            transition: {
              duration: 0.6,
              ease: "easeOut"
            }
          },
          visible: { 
            opacity: 1,
            x: 0,
            transition: {
              duration: 0.6,
              ease: "easeOut",
              staggerChildren: 0.1
            }
          },
          exit: { 
            opacity: 0,
            x: -100,
            transition: {
              duration: 0.4,
              ease: "easeIn"
            }
          }
        };
      case "right":
        return {
          hidden: { 
            opacity: 0,
            x: -100,
            transition: {
              duration: 0.6,
              ease: "easeOut"
            }
          },
          visible: { 
            opacity: 1,
            x: 0,
            transition: {
              duration: 0.6,
              ease: "easeOut",
              staggerChildren: 0.1
            }
          },
          exit: { 
            opacity: 0,
            x: 100,
            transition: {
              duration: 0.4,
              ease: "easeIn"
            }
          }
        };
      case "up":
        return {
          hidden: { 
            opacity: 0,
            y: 100,
            transition: {
              duration: 0.6,
              ease: "easeOut"
            }
          },
          visible: { 
            opacity: 1,
            y: 0,
            transition: {
              duration: 0.6,
              ease: "easeOut",
              staggerChildren: 0.1
            }
          },
          exit: { 
            opacity: 0,
            y: -100,
            transition: {
              duration: 0.4,
              ease: "easeIn"
            }
          }
        };
      case "down":
        return {
          hidden: { 
            opacity: 0,
            y: -100,
            transition: {
              duration: 0.6,
              ease: "easeOut"
            }
          },
          visible: { 
            opacity: 1,
            y: 0,
            transition: {
              duration: 0.6,
              ease: "easeOut",
              staggerChildren: 0.1
            }
          },
          exit: { 
            opacity: 0,
            y: 100,
            transition: {
              duration: 0.4,
              ease: "easeIn"
            }
          }
        };
      default:
        return baseVariants;
    }
  };

  return (
    <AnimatePresence mode="wait">
      {isVisible && (
        <motion.div
          key={direction}
          variants={getVariants()}
          initial="hidden"
          animate="visible"
          exit="exit"
          className={className}
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
