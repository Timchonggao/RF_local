# **Development**

## Introduction

<!--
近年来，辐射场技术的爆炸性发展使得本领域下涌现出了大量优秀的学术工作，然而，囿于整理内卷化的学术环境和重论文、轻工程的普遍风气，大量的工作缺乏良好维护的开源代码实现，难以阅读和扩展，为基于这些工作的后续研究带来了阻碍。

以nerfstudio为代表，一些注意到这种现象的研究者为此编写了功能强大、组件完善的代码库，试图搭建统一的代码框架，一方面使得不同的技术在实现上能基于相同的基础框架，便于相互扩展以及减少学习者的心智负担，另一方面，这也大大降低了实现新技术的工程难度，使得研究者能够将更多精力放在理论方法本身上。

然而，随着代码库体量的增大，这一类代码库几乎不可避免地衍生出一些问题，例如极端复杂的继承结构、面向对象编程带来的过度抽象。这些问题使得初学者在阅读学习代码库时往往面临极其陡峭的学习曲线，在实现新方法时，更是容易遭遇框架灵活性严重不足的问题。

本代码库的作者饱受其苦，决心开发一份代码整洁、学习曲线平滑、易于扩展、易于维护的代码库，试图在简洁性、灵活性、功能性这个不可能三角中取得最好的平衡。
-->

In recent years, the explosive development of radiance field technology has led to the emergence of numerous outstanding academic works in this field. However, due to the competitive academic environment and the prevalent focus on publishing papers over engineering practices, many works lack well-maintained open-source code implementations, making them difficult to read and extend. This hinders subsequent research based on these works.

Researchers, aware of this issue, have developed powerful and well-structured code libraries, such as [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), aiming to establish a unified code framework. This approach not only allows different technologies to be implemented based on a common framework, facilitating mutual extension and reducing the cognitive load for learners, but also significantly lowers the engineering difficulty of implementing new technologies, enabling researchers to focus more on theoretical methods themselves.

However, as these code libraries grow in size, they almost inevitably develop certain issues, such as extremely complex inheritance structures and over-abstraction brought by object-oriented programming. These problems often result in a steep learning curve for beginners when reading and learning from the code library, and a significant lack of flexibility when implementing new methods.

The author of this code library has suffered from these issues and is determined to develop a code library that is clean, has a smooth learning curve, and is easy to extend and maintain. The goal is to achieve the best balance among simplicity, flexibility, and functionality in this challenging trade-off.

## Overview

<!-- 因此，本教程旨在提供一份由浅入深的学习指导，将围绕代码库的设计哲学，首先从宏观视角解释如何在简洁性、灵活性、功能性之中取得平衡，并抽丝剥茧地逐步介绍代码库所依赖的关键组件或语言特性，包括[类型注解](./type-hints.md)、[数据结构](./data-structure.md)和[代码架构](./code-architecture.md)；随后，本教程将从[方法实现](./method-implementation.md)、[数据集实现](./dataset-implementation.md)两方面进行详细细节的指导，作为扩展本代码库的示例。 -->

Therefore, this tutorial aims to provide a step-by-step learning guide, starting from basic to advanced concepts. It will revolve around the design philosophy of the code library, initially explaining how to achieve a balance among simplicity, flexibility, and functionality from a macro perspective. The tutorial will then gradually introduce the key components or language features that the code library relies on, including [type hints](./type-hints.md), [data structures](./data-structure.md), and [code architecture](./code-architecture.md). Subsequently, the tutorial will provide detailed guidance on [method implementation](./method-implementation.md) and [dataset implementation](./dataset-implementation.md), offering examples on how to extend the code library.

## Type Hints

<!-- 众所周知，Python并非一门静态类型语言，其便捷性很大一部分便来自于Python代码中不需要显式指定变量类型的特性。然而随着2014年草案[PEP 484](https://peps.python.org/pep-0484/)的提出并得到采纳，类型注解作为一种内置功能，正式加入了Python 3.5中。类型注解有诸多好处，其中最受青睐的一条在于类型注解能够增强代码的静态类型检查，从而在运行前就能提前捕捉代码的潜在错误（正如一系列静态类型语言那样）。然而，本代码库却处于另一个重要原因而采用类型注解，那就是类型注解不但能够使得代码的阅读者可以采用分解的方式来理解代码从而避免过度的思维堆栈，更能允许现代的代码编辑器对代码进行更好的分析和提示。

关于此部分的详细教程，请参见[类型注解](./type-hints.md)。 -->

It is well known that Python is not a statically typed language, and much of its convenience comes from the feature that explicit type specification for variables is not required in Python code. However, with the proposal and adoption of the 2014 draft [PEP 484](https://peps.python.org/pep-0484/), type annotations were officially introduced as a built-in feature in Python 3.5. 

Type annotations offer many benefits, one of the most favored being their ability to enhance static type checking, thereby catching potential errors before runtime, much like other statically typed languages. However, our code library adopts type annotations for another important reason: they not only allow code readers to understand the code in a decomposed manner, avoiding excessive cognitive load, but also enable modern code editors to perform better analysis and provide more accurate suggestions.

For a detailed tutorial on this part, please refer to [Type Hints](./type-hints.md).

## Data Structure


